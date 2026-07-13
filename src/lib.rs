use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::ptr;
use std::slice;
use std::sync::OnceLock;

const BOARD: i8 = 9;
const ANCHORS: i8 = 8;
const ACTIONS: usize = 140;
const STATE_BYTES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct State {
    horizontal: u64,
    vertical: u64,
    pawns: [u8; 2],
    walls: [u8; 2],
    current: u8,
    ply: u8,
}

impl State {
    fn decode(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() != STATE_BYTES {
            return Err("state key must contain exactly 20 bytes");
        }
        let horizontal = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let vertical = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let packed_pawns = u16::from_le_bytes(data[16..18].try_into().unwrap());
        let pawns = [
            (packed_pawns & 0x7f) as u8,
            ((packed_pawns >> 7) & 0x7f) as u8,
        ];
        let current = ((packed_pawns >> 14) & 1) as u8;
        let walls = [data[18] & 0x0f, data[18] >> 4];
        if pawns[0] >= 81 || pawns[1] >= 81 || pawns[0] == pawns[1] {
            return Err("state contains invalid pawn positions");
        }
        if walls[0] > 10 || walls[1] > 10 {
            return Err("state contains an invalid wall count");
        }
        Ok(Self {
            horizontal,
            vertical,
            pawns,
            walls,
            current,
            ply: data[19],
        })
    }

    fn encode(self) -> [u8; STATE_BYTES] {
        let mut data = [0; STATE_BYTES];
        data[0..8].copy_from_slice(&self.horizontal.to_le_bytes());
        data[8..16].copy_from_slice(&self.vertical.to_le_bytes());
        let packed_pawns =
            self.pawns[0] as u16 | ((self.pawns[1] as u16) << 7) | ((self.current as u16) << 14);
        data[16..18].copy_from_slice(&packed_pawns.to_le_bytes());
        data[18] = self.walls[0] | (self.walls[1] << 4);
        data[19] = self.ply;
        data
    }

    fn position(self, player: usize) -> (i8, i8) {
        (
            (self.pawns[player] / 9) as i8,
            (self.pawns[player] % 9) as i8,
        )
    }

    fn terminal(self, max_plies: u8) -> u8 {
        let previous = 1 - self.current as usize;
        let (row, _) = self.position(previous);
        if (previous == 0 && row == 8) || (previous == 1 && row == 0) {
            1 // mover has lost: the opponent just reached its goal
        } else if self.ply >= max_plies {
            2 // capped
        } else {
            0
        }
    }
}

fn bit(row: i8, col: i8) -> u64 {
    1u64 << (row as u64 * 8 + col as u64)
}

fn in_board(row: i8, col: i8) -> bool {
    (0..BOARD).contains(&row) && (0..BOARD).contains(&col)
}

fn can_cross(state: State, from: (i8, i8), to: (i8, i8)) -> bool {
    if !in_board(from.0, from.1) || !in_board(to.0, to.1) {
        return false;
    }
    let (dr, dc) = (to.0 - from.0, to.1 - from.1);
    if dr.abs() + dc.abs() != 1 {
        return false;
    }
    if dr != 0 {
        let boundary_row = from.0.min(to.0);
        let col = from.1;
        let left = col > 0 && state.horizontal & bit(boundary_row, col - 1) != 0;
        let right = col < ANCHORS && state.horizontal & bit(boundary_row, col) != 0;
        !(left || right)
    } else {
        let row = from.0;
        let boundary_col = from.1.min(to.1);
        let above = row > 0 && state.vertical & bit(row - 1, boundary_col) != 0;
        let below = row < ANCHORS && state.vertical & bit(row, boundary_col) != 0;
        !(above || below)
    }
}

fn canonical_delta(action: usize, player: u8) -> Option<(i8, i8)> {
    let (mut dr, dc) = match action {
        0 | 4 => (1, 0),
        1 | 5 => (-1, 0),
        2 | 6 => (0, 1),
        3 | 7 => (0, -1),
        8 => (1, 1),
        9 => (1, -1),
        10 => (-1, 1),
        11 => (-1, -1),
        _ => return None,
    };
    if player == 1 {
        dr = -dr;
    }
    Some((dr, dc))
}

fn move_destination(state: State, action: usize) -> Option<(i8, i8)> {
    if action >= 12 {
        return None;
    }
    let player = state.current as usize;
    let pawn = state.position(player);
    let opponent = state.position(1 - player);
    let delta = canonical_delta(action, state.current)?;

    if action < 4 {
        let target = (pawn.0 + delta.0, pawn.1 + delta.1);
        return (target != opponent && can_cross(state, pawn, target)).then_some(target);
    }
    if action < 8 {
        let adjacent = (pawn.0 + delta.0, pawn.1 + delta.1);
        let landing = (adjacent.0 + delta.0, adjacent.1 + delta.1);
        return (adjacent == opponent
            && can_cross(state, pawn, adjacent)
            && can_cross(state, adjacent, landing))
        .then_some(landing);
    }

    let adjacent_delta = (opponent.0 - pawn.0, opponent.1 - pawn.1);
    if adjacent_delta.0.abs() + adjacent_delta.1.abs() != 1 || !can_cross(state, pawn, opponent) {
        return None;
    }
    let straight = (opponent.0 + adjacent_delta.0, opponent.1 + adjacent_delta.1);
    if can_cross(state, opponent, straight) {
        return None;
    }
    let target = (pawn.0 + delta.0, pawn.1 + delta.1);
    (in_board(target.0, target.1)
        && (target.0 - opponent.0).abs() + (target.1 - opponent.1).abs() == 1
        && can_cross(state, opponent, target))
    .then_some(target)
}

fn shortest_path(state: State, player: usize) -> Option<u8> {
    let start = state.pawns[player] as usize;
    let goal = if player == 0 { 8 } else { 0 };
    let mut seen = [false; 81];
    let mut queue = VecDeque::with_capacity(81);
    seen[start] = true;
    queue.push_back((start, 0u8));
    while let Some((cell, distance)) = queue.pop_front() {
        let row = (cell / 9) as i8;
        let col = (cell % 9) as i8;
        if row == goal {
            return Some(distance);
        }
        for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let next = (row + dr, col + dc);
            if !can_cross(state, (row, col), next) {
                continue;
            }
            let index = (next.0 * 9 + next.1) as usize;
            if !seen[index] {
                seen[index] = true;
                queue.push_back((index, distance + 1));
            }
        }
    }
    None
}

fn path_exists(state: State, player: usize) -> bool {
    let start = state.pawns[player] as usize;
    let goal = if player == 0 { 8i8 } else { 0i8 };
    let start_row = (start / 9) as i8;
    let mut best = [u8::MAX; 81];
    let mut open = BinaryHeap::with_capacity(81);
    best[start] = 0;
    open.push(Reverse(((goal - start_row).unsigned_abs(), 0u8, start)));
    while let Some(Reverse((_, distance, cell))) = open.pop() {
        if distance != best[cell] {
            continue;
        }
        let row = (cell / 9) as i8;
        let col = (cell % 9) as i8;
        if row == goal {
            return true;
        }
        for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let next = (row + dr, col + dc);
            if !can_cross(state, (row, col), next) {
                continue;
            }
            let index = (next.0 * 9 + next.1) as usize;
            let next_distance = distance + 1;
            if next_distance < best[index] {
                best[index] = next_distance;
                let estimate = next_distance + (goal - next.0).unsigned_abs();
                open.push(Reverse((estimate, next_distance, index)));
            }
        }
    }
    false
}

type ConflictMasks = [(u64, u64); 64];
static HORIZONTAL_CONFLICTS: OnceLock<ConflictMasks> = OnceLock::new();
static VERTICAL_CONFLICTS: OnceLock<ConflictMasks> = OnceLock::new();

fn conflict_masks(horizontal: bool) -> &'static ConflictMasks {
    let cell = if horizontal {
        &HORIZONTAL_CONFLICTS
    } else {
        &VERTICAL_CONFLICTS
    };
    cell.get_or_init(|| {
        std::array::from_fn(|index| {
            let row = (index / 8) as i8;
            let col = (index % 8) as i8;
            let mut same = bit(row, col);
            if horizontal {
                if col > 0 {
                    same |= bit(row, col - 1);
                }
                if col + 1 < ANCHORS {
                    same |= bit(row, col + 1);
                }
            } else {
                if row > 0 {
                    same |= bit(row - 1, col);
                }
                if row + 1 < ANCHORS {
                    same |= bit(row + 1, col);
                }
            }
            (same, bit(row, col))
        })
    })
}

fn wall_legal(state: State, horizontal: bool, row: i8, col: i8) -> bool {
    if state.walls[state.current as usize] == 0
        || !(0..ANCHORS).contains(&row)
        || !(0..ANCHORS).contains(&col)
    {
        return false;
    }
    let index = (row * 8 + col) as usize;
    let candidate = bit(row, col);
    let same = if horizontal {
        state.horizontal
    } else {
        state.vertical
    };
    let crossing = if horizontal {
        state.vertical
    } else {
        state.horizontal
    };
    let (same_conflicts, crossing_conflicts) = conflict_masks(horizontal)[index];
    if same & same_conflicts != 0 || crossing & crossing_conflicts != 0 {
        return false;
    }
    let mut candidate_state = state;
    if horizontal {
        candidate_state.horizontal |= candidate;
    } else {
        candidate_state.vertical |= candidate;
    }
    path_exists(candidate_state, 0) && path_exists(candidate_state, 1)
}

fn wall_from_action(state: State, action: usize) -> (bool, i8, i8, usize) {
    let (horizontal, canonical_index) = if action < 76 {
        (true, action - 12)
    } else {
        (false, action - 76)
    };
    let canonical_row = canonical_index / 8;
    let col = canonical_index % 8;
    let row = if state.current == 1 {
        7 - canonical_row
    } else {
        canonical_row
    };
    (horizontal, row as i8, col as i8, row * 8 + col)
}

fn legal_action(state: State, action: usize) -> bool {
    match action {
        0..=11 => move_destination(state, action).is_some(),
        12..=139 => {
            let (horizontal, row, col, _) = wall_from_action(state, action);
            wall_legal(state, horizontal, row, col)
        }
        _ => false,
    }
}

fn apply_legal_action(mut state: State, action: usize) -> State {
    let player = state.current as usize;
    if action < 12 {
        let (row, col) = move_destination(state, action).unwrap();
        state.pawns[player] = (row * 9 + col) as u8;
    } else {
        let (horizontal, _, _, index) = wall_from_action(state, action);
        let wall_bit = 1u64 << index;
        if horizontal {
            state.horizontal |= wall_bit;
        } else {
            state.vertical |= wall_bit;
        }
        state.walls[player] -= 1;
    }
    state.current = 1 - state.current;
    state.ply = state.ply.saturating_add(1);
    state
}

fn successor(state: State, action: usize, max_plies: u8) -> Result<State, i32> {
    if state.terminal(max_plies) != 0 {
        return Err(2);
    }
    if !legal_action(state, action) {
        return Err(3);
    }
    Ok(apply_legal_action(state, action))
}

fn mirrored_state(state: State) -> State {
    fn mirror_walls(walls: u64) -> u64 {
        let mut mirrored = 0u64;
        for index in 0..64 {
            if walls & (1u64 << index) != 0 {
                let row = index / 8;
                let col = index % 8;
                mirrored |= 1u64 << (row * 8 + 7 - col);
            }
        }
        mirrored
    }

    let mut mirrored = state;
    mirrored.horizontal = mirror_walls(state.horizontal);
    mirrored.vertical = mirror_walls(state.vertical);
    for player in 0..2 {
        let row = state.pawns[player] / 9;
        let col = state.pawns[player] % 9;
        mirrored.pawns[player] = row * 9 + 8 - col;
    }
    mirrored
}

fn perft_key(state: State, depth: u8) -> ([u8; STATE_BYTES], u8) {
    let original = state.encode();
    let mirrored = mirrored_state(state).encode();
    (original.min(mirrored), depth)
}

fn perft_cached(
    state: State,
    depth: u8,
    max_plies: u8,
    cache: &mut HashMap<([u8; STATE_BYTES], u8), u64>,
) -> u64 {
    if depth == 0 {
        return 1;
    }
    if state.terminal(max_plies) != 0 {
        return 0;
    }
    let key = perft_key(state, depth);
    if let Some(&cached) = cache.get(&key) {
        return cached;
    }
    let result = if depth == 1 {
        (0..ACTIONS)
            .filter(|&action| legal_action(state, action))
            .count() as u64
    } else {
        let mut nodes = 0u64;
        for action in 0..ACTIONS {
            if legal_action(state, action) {
                nodes += perft_cached(
                    apply_legal_action(state, action),
                    depth - 1,
                    max_plies,
                    cache,
                );
            }
        }
        nodes
    };
    cache.insert(key, result);
    result
}

/// Fill a 140-byte boolean mask. Returns 0 on success and 1 for invalid input.
#[no_mangle]
pub unsafe extern "C" fn br_legal_actions(
    state_data: *const u8,
    max_plies: u8,
    output: *mut u8,
) -> i32 {
    if state_data.is_null() || output.is_null() {
        return 1;
    }
    let data = slice::from_raw_parts(state_data, STATE_BYTES);
    let Ok(state) = State::decode(data) else {
        return 1;
    };
    for action in 0..ACTIONS {
        *output.add(action) =
            u8::from(state.terminal(max_plies) == 0 && legal_action(state, action));
    }
    0
}

/// Write a successor state. Returns 0, or 1 invalid state, 2 terminal, 3 illegal action.
#[no_mangle]
pub unsafe extern "C" fn br_next_state(
    state_data: *const u8,
    action: u16,
    max_plies: u8,
    output: *mut u8,
) -> i32 {
    if state_data.is_null() || output.is_null() {
        return 1;
    }
    let data = slice::from_raw_parts(state_data, STATE_BYTES);
    let Ok(state) = State::decode(data) else {
        return 1;
    };
    match successor(state, action as usize, max_plies) {
        Ok(next) => {
            ptr::copy_nonoverlapping(next.encode().as_ptr(), output, STATE_BYTES);
            0
        }
        Err(code) => code,
    }
}

/// Return terminal status 0..2, or -1 for invalid input.
#[no_mangle]
pub unsafe extern "C" fn br_terminal_status(state_data: *const u8, max_plies: u8) -> i32 {
    if state_data.is_null() {
        return -1;
    }
    let data = slice::from_raw_parts(state_data, STATE_BYTES);
    State::decode(data)
        .map(|state| state.terminal(max_plies) as i32)
        .unwrap_or(-1)
}

/// Return distance, -1 if unreachable, or -2 for invalid input/player.
#[no_mangle]
pub unsafe extern "C" fn br_shortest_path_distance(state_data: *const u8, player: u8) -> i32 {
    if state_data.is_null() || player > 1 {
        return -2;
    }
    let data = slice::from_raw_parts(state_data, STATE_BYTES);
    let Ok(state) = State::decode(data) else {
        return -2;
    };
    shortest_path(state, player as usize)
        .map(i32::from)
        .unwrap_or(-1)
}

/// Count legal leaf move sequences. Returns 0 on success and 1 for invalid input.
#[no_mangle]
pub unsafe extern "C" fn br_perft(
    state_data: *const u8,
    depth: u8,
    max_plies: u8,
    output: *mut u64,
) -> i32 {
    if state_data.is_null() || output.is_null() {
        return 1;
    }
    let data = slice::from_raw_parts(state_data, STATE_BYTES);
    let Ok(state) = State::decode(data) else {
        return 1;
    };
    let mut cache = HashMap::new();
    *output = perft_cached(state, depth, max_plies, &mut cache);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial() -> State {
        State {
            horizontal: 0,
            vertical: 0,
            pawns: [4, 76],
            walls: [10, 10],
            current: 0,
            ply: 0,
        }
    }

    #[test]
    fn packed_state_is_exactly_twenty_bytes_and_round_trips() {
        let state = initial();
        let encoded = state.encode();
        assert_eq!(encoded.len(), 20);
        assert_eq!(State::decode(&encoded), Ok(state));
    }

    #[test]
    fn initial_move_count_is_stable() {
        let state = initial();
        assert_eq!(
            (0..ACTIONS).filter(|&a| legal_action(state, a)).count(),
            131
        );
    }
}
