"""
game.py — Tsoro Yematatu Game Environment
==========================================
Tsoro Yematatu is a two-player abstract strategy game from Zimbabwe.
Players place then move 3 pieces on a 7-point board to form a 3-in-a-row.

Board layout (7 positions):
         0
        / \
       1   2
      / \ / \
     3   4   5
          \  |
           \ |
            6  (centre bottom)

Valid lines (winning combinations):
    0-1-3, 0-2-5, 1-4-5, 3-4-2, 0-4-6, 1-2-6  <-- adjusted per actual board
"""

# ──────────────────────────────────────────────
# Board definition
# ──────────────────────────────────────────────

# The 7 positions are numbered 0-6.
# Adjacency: which positions a piece can move to in one step.
ADJACENCY = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4, 5],
    3: [1, 4],
    4: [1, 2, 3, 5, 6],
    5: [2, 4],
    6: [3, 4, 5],
}

# All straight lines on the board (winning combinations).
WINNING_LINES = [
    (0, 1, 3),   # top-left diagonal
    (0, 2, 5),   # top-right diagonal
    (3, 4, 5),   # bottom row
    (0, 4, 6),   # centre vertical
    (1, 4, 5),   # middle-left to bottom-right
    (2, 4, 3),   # middle-right to bottom-left
    (1, 2, 6),   # top-middle row
    (3, 6, 5),   # bottom horizontal
]

PLAYER1 = 1
PLAYER2 = 2
EMPTY   = 0

PHASE_PLACEMENT = "placement"
PHASE_MOVEMENT  = "movement"


class TsoroYematatu:
    """
    The game environment.

    board       : list of 7 integers (0=empty, 1=player1, 2=player2)
    phase       : 'placement' or 'movement'
    turn        : 1 or 2 (whose turn it is)
    pieces_placed: dict tracking how many pieces each player has placed
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board to the starting state. Returns the initial state."""
        self.board          = [EMPTY] * 7
        self.phase          = PHASE_PLACEMENT
        self.turn           = PLAYER1
        self.pieces_placed  = {PLAYER1: 0, PLAYER2: 0}
        self.done           = False
        self.winner         = None
        return self.get_state()

    # ──────────────────────────────────────────────
    # State representation
    # ──────────────────────────────────────────────

    def get_state(self):
        """
        Returns a hashable tuple representing the full game state:
        (board_tuple, phase, turn)
        Used as the key in the Q-table.
        """
        return (tuple(self.board), self.phase, self.turn)

    # ──────────────────────────────────────────────
    # Legal moves
    # ──────────────────────────────────────────────

    def get_legal_moves(self):
        """
        Returns a list of legal moves for the current player.

        Placement phase:  move = position index (0-6) to place a piece
        Movement phase:   move = (from_pos, to_pos) tuple
        """
        if self.done:
            return []

        if self.phase == PHASE_PLACEMENT:
            # Any empty position is a valid placement
            return [i for i, cell in enumerate(self.board) if cell == EMPTY]

        else:  # PHASE_MOVEMENT
            moves = []
            for pos in range(7):
                if self.board[pos] == self.turn:
                    # Regular adjacency moves
                    for dest in ADJACENCY[pos]:
                        if self.board[dest] == EMPTY:
                            moves.append((pos, dest))
                    # Jump moves: jump over any adjacent piece to the empty spot beyond
                    for mid in ADJACENCY[pos]:
                        # Find positions adjacent to 'mid' that are in line with pos→mid
                        for dest in ADJACENCY[mid]:
                            if dest != pos and self.board[dest] == EMPTY:
                                # Check it's a straight-line jump (dest is collinear)
                                if self._is_collinear(pos, mid, dest):
                                    moves.append((pos, dest))
            return moves

    def _is_collinear(self, a, b, c):
        """Check if three positions lie on one of the defined winning lines."""
        triple = tuple(sorted([a, b, c]))
        for line in WINNING_LINES:
            if tuple(sorted(line)) == triple:
                return True
        return False

    # ──────────────────────────────────────────────
    # Apply a move
    # ──────────────────────────────────────────────

    def step(self, move):
        """
        Apply a move and advance the game state.

        Returns:
            state   : new game state (tuple)
            reward  : +1 win, -1 loss, 0 otherwise (from current player's perspective)
            done    : True if the game has ended
            info    : dict with extra info (winner, etc.)
        """
        if self.done:
            raise ValueError("Game is already over. Call reset() to start a new game.")

        legal = self.get_legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move}. Legal moves: {legal}")

        current_player = self.turn

        # ── Apply the move ──
        if self.phase == PHASE_PLACEMENT:
            self.board[move] = current_player
            self.pieces_placed[current_player] += 1

            # Switch to movement phase once all 6 pieces are placed
            if self.pieces_placed[PLAYER1] == 3 and self.pieces_placed[PLAYER2] == 3:
                self.phase = PHASE_MOVEMENT

        else:  # PHASE_MOVEMENT
            from_pos, to_pos = move
            self.board[from_pos] = EMPTY
            self.board[to_pos]   = current_player

        # ── Check for a winner ──
        reward = 0
        if self._check_winner(current_player):
            self.done   = True
            self.winner = current_player
            reward      = 1  # reward for the player who just moved

        # ── Check for draw (no legal moves for next player) ──
        self.turn = PLAYER2 if current_player == PLAYER1 else PLAYER1
        if not self.done:
            if not self.get_legal_moves():
                self.done   = True
                self.winner = None  # draw
                reward      = 0

        return self.get_state(), reward, self.done, {"winner": self.winner}

    # ──────────────────────────────────────────────
    # Win detection
    # ──────────────────────────────────────────────

    def _check_winner(self, player):
        """Return True if 'player' has formed a 3-in-a-row."""
        for line in WINNING_LINES:
            if all(self.board[pos] == player for pos in line):
                return True
        return False

    # ──────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────

    def render(self):
        """Print a simple ASCII representation of the board."""
        s = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        b = self.board
        print(f"\n    {s[b[0]]}")
        print(f"   / \\")
        print(f"  {s[b[1]]}   {s[b[2]]}")
        print(f" / \\ / \\")
        print(f"{s[b[3]]}   {s[b[4]]}   {s[b[5]]}")
        print(f"     \\  |")
        print(f"      {s[b[6]]}")
        print(f"\nPhase: {self.phase} | Turn: Player {self.turn}")
        print(f"Legal moves: {self.get_legal_moves()}\n")
