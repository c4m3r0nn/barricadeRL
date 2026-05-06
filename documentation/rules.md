Core environment rules
Players
Two-player zero-sum game.
One pawn per player.
Players alternate turns.
Environment state is always from the perspective of the current player when training, unless using an absolute board encoding.
Board
Use a 9×9 grid of pawn squares.
Player 0 starts at the center of one edge: (8, 4) if moving upward.
Player 1 starts at the opposite center: (0, 4) if moving downward.
Each player’s goal is any square on the opposite edge.
Walls / barricades
Each player starts with 10 walls.
A wall is one continuous piece that spans two adjacent square-border edges.
Walls may be horizontal or vertical.
Walls have a length of two. A single wall has no gap in its middle.
Two same-orientation walls cannot overlap one of their two occupied edges, so adjacent anchors in the same line are illegal.
If two same-orientation walls are placed end-to-end without overlapping, a small endpoint gap remains between the two separate wall pieces. A perpendicular wall may be placed through that gap, making a plus-shaped junction.
On a 9×9 board, valid wall anchors are usually an 8×8 grid.
Total possible wall placements:
64 horizontal wall anchors.
64 vertical wall anchors.
128 wall actions total.
Once placed, a wall remains for the rest of the game.
Turn structure
On each turn, the active player chooses exactly one action:
Move pawn.
Place wall.
If the player has no walls remaining, only pawn moves are legal.
After a legal action, control passes to the opponent.
Pawn movement
A pawn normally moves one square:
Up.
Down.
Left.
Right.
Pawns cannot move diagonally under normal movement.
Pawns cannot move through walls.
Pawns cannot move off the board.
Pawns cannot occupy the same square.
Jumping rules
If the opponent pawn is directly adjacent and no wall separates the two pawns, the active pawn may jump over the opponent.
A straight jump is legal if the square behind the opponent is:
On the board.
Not blocked by a wall.
If a straight jump is blocked by a wall or board edge, diagonal side-jumps around the opponent are legal when not blocked.
Jump moves should be represented as legal move actions generated dynamically, not as free diagonal moves.
Wall placement legality
A wall cannot overlap an existing wall.
A wall cannot cross another wall.
A wall cannot extend outside the board.
A wall cannot be placed if the player has zero walls remaining.
A wall cannot completely block either player from reaching their goal row.
To validate this, run a shortest-path or reachability check after hypothetical wall placement.
Path-preservation rule
After every wall placement, both players must still have at least one valid path to their goal.
Implement this with BFS, A*, or DFS over the 9×9 pawn graph.
Reject wall placements that make either player’s goal unreachable.
Terminal condition
The game ends immediately when a player’s pawn reaches any square on their target row.
The reaching player wins.
No additional turn is given to the opponent.
Action-space rules
Recommended discrete action space
4 base pawn moves:
Move up.
Move down.
Move left.
Move right.
128 wall placements:
64 horizontal.
64 vertical.
Total nominal action count: 132.
Jump moves can be handled by mapping the four directional move actions to context-dependent legal destinations.
Alternative action space
Include explicit jump and diagonal-jump actions:
4 normal directions.
4 straight jumps.
4 diagonal jump options.
128 wall actions.
This is easier to debug but less compact.
Action masking
The environment should return a legal-action mask every step.
Illegal moves should not be sampled during training when using policy-gradient or tree-search agents.
For algorithms that cannot use masks, illegal actions can:
Receive a penalty.
Become a no-op.
End the episode as a loss.
Best practice: use masking rather than teaching legality through punishment.
Observation/state rules
Minimum state representation
Current player pawn position.
Opponent pawn position.
Horizontal wall grid.
Vertical wall grid.
Current player walls remaining.
Opponent walls remaining.
Side to move, if using absolute rather than canonical state.
Recommended tensor encoding
Plane 1: current player pawn location, 9×9.
Plane 2: opponent pawn location, 9×9.
Plane 3: horizontal wall anchors, padded to 9×9.
Plane 4: vertical wall anchors, padded to 9×9.
Plane 5: current player walls remaining, constant plane.
Plane 6: opponent walls remaining, constant plane.
Optional plane: legal moves mask or shortest-path features.
Canonical perspective
For self-play, encode the board from the current player’s perspective.
Rotate or mirror the board so the active player is always moving “up.”
Swap wall counts and pawn planes accordingly.
This improves sample efficiency by exploiting game symmetry.
Reward rules
Sparse reward
Win: +1.
Loss: -1.
Non-terminal step: 0.
Optional shaped reward
Small reward for reducing your shortest path to goal.
Small penalty if opponent’s shortest path becomes shorter.
Small penalty per turn to encourage faster wins.
Keep shaped rewards small so the model still optimizes winning, not just path length.
Example shaped reward
reward = terminal_reward
plus 0.01 * (previous_my_shortest_path - current_my_shortest_path)
minus 0.01 * (previous_opp_shortest_path - current_opp_shortest_path)
optional step penalty: -0.001.
Step-function rules
On reset()
Clear all walls.
Place pawns at starting squares.
Set wall counts to 10 each.
Set current player.
Return initial observation and legal-action mask.
On step(action)
Check if action is legal.
If illegal, handle according to your invalid-action policy.
If legal pawn move:
Update pawn position.
If legal wall placement:
Add wall.
Decrement active player’s wall count.
Check win condition.
Compute reward.
Swap active player if non-terminal.
Return observation, reward, terminal flag, truncation flag, and info.
On terminal
Do not allow further moves.
Return final reward from the perspective of the player who acted, or normalize rewards consistently depending on your RL library.
Environment correctness tests
Movement tests
Pawn cannot move through walls.
Pawn cannot move off board.
Pawn cannot move onto opponent’s square.
Pawn can move backward and sideways.
Pawn can jump over adjacent opponent when legal.
Pawn can side-jump when straight jump is blocked.
Wall tests
Wall count decreases after placement.
Wall cannot overlap existing wall.
Wall cannot cross existing wall.
Wall cannot be placed outside board.
Wall cannot be placed when no walls remain.
Wall cannot eliminate all paths for either player.
Win tests
Player wins immediately on reaching the opposite row.
Player does not need to reach a specific square, only the goal row.
Environment terminates after win.
Symmetry tests
Mirrored or rotated equivalent states should produce equivalent legal-action masks.
Canonical current-player encoding should preserve action semantics.
