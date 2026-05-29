"""
game.py — Tsoro Yematatu Game Environment
==========================================
Tsoro Yematatu is a two-player abstract strategy game from Zimbabwe.
Players place then move 3 pieces on a 7-point board to form a 3-in-a-row.

Board layout (7 positions):

         0
        /|\
       / | \
      1--2--3
     /   |   \
    4----5----6

  0 = top apex
  1 = middle-left,   2 = middle-centre,  3 = middle-right
  4 = bottom-left,   5 = bottom-centre,  6 = bottom-right

Adjacency (connected by lines):
  0: 1, 2, 3
  1: 0, 2, 4
  2: 0, 1, 3, 5
  3: 0, 2, 6
  4: 1, 5
  5: 2, 4, 6
  6: 3, 5

Winning lines (3-in-a-row along any straight line):
  0-1-4  (left diagonal)
  0-3-6  (right diagonal)
  0-2-5  (centre vertical)
  1-2-3  (middle horizontal)
  4-5-6  (bottom horizontal)
  1-5-6  ... etc — see WINNING_LINES below
"""

ADJACENCY = {
    0: [1, 2, 3],
    1: [0, 2, 4],
    2: [0, 1, 3, 5],
    3: [0, 2, 6],
    4: [1, 5],
    5: [2, 4, 6],
    6: [3, 5],
}

WINNING_LINES = [
    (0, 1, 4),   # left outer diagonal
    (0, 3, 6),   # right outer diagonal
    (0, 2, 5),   # centre vertical
    (1, 2, 3),   # middle horizontal
    (4, 5, 6),   # bottom horizontal
    (1, 5, 6),   # middle-left to bottom-right cross
    (3, 2, 4),   # middle-right to bottom-left cross (through centre)
]

PLAYER1 = 1
PLAYER2 = 2
EMPTY   = 0

PHASE_PLACEMENT = "placement"
PHASE_MOVEMENT  = "movement"


class TsoroYematatu:
    """
    The game environment.

    board        : list of 7 integers (0=empty, 1=player1, 2=player2)
    phase        : 'placement' or 'movement'
    turn         : 1 or 2 (whose turn it is)
    pieces_placed: dict tracking how many pieces each player has placed
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board to the starting state. Returns the initial state."""
        self.board         = [EMPTY] * 7
        self.phase         = PHASE_PLACEMENT
        self.turn          = PLAYER1
        self.pieces_placed = {PLAYER1: 0, PLAYER2: 0}
        self.done          = False
        self.winner        = None
        return self.get_state()

    def get_state(self):
        """
        Returns a hashable tuple representing the full game state:
        (board_tuple, phase, turn)
        Used as the key in the Q-table.
        """
        return (tuple(self.board), self.phase, self.turn)

    def get_legal_moves(self):
        """
        Returns a list of legal moves for the current player.

        Placement phase : move = position index (0-6) to place a piece
        Movement phase  : move = (from_pos, to_pos) tuple
        """
        if self.done:
            return []

        if self.phase == PHASE_PLACEMENT:
            return [i for i, cell in enumerate(self.board) if cell == EMPTY]

        else:
            moves = []
            for pos in range(7):
                if self.board[pos] == self.turn:
                    for dest in ADJACENCY[pos]:
                        if self.board[dest] == EMPTY:
                            moves.append((pos, dest))
                    for mid in ADJACENCY[pos]:
                        for dest in ADJACENCY[mid]:
                            if dest != pos and self.board[dest] == EMPTY:
                                if self._is_collinear(pos, mid, dest):
                                    moves.append((pos, dest))
            return list(set(moves))

    def _is_collinear(self, a, b, c):
        """Check if three positions lie on one of the defined winning lines."""
        triple = tuple(sorted([a, b, c]))
        for line in WINNING_LINES:
            if tuple(sorted(line)) == triple:
                return True
        return False

    def step(self, move):
        """
        Apply a move and advance the game state.

        Returns:
            state  : new game state (tuple)
            reward : +1 win, -1 loss, 0 otherwise
            done   : True if the game has ended
            info   : dict with winner
        """
        if self.done:
            raise ValueError("Game is already over. Call reset().")

        legal = self.get_legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move}. Legal: {legal}")

        current_player = self.turn

        if self.phase == PHASE_PLACEMENT:
            self.board[move] = current_player
            self.pieces_placed[current_player] += 1
            if self.pieces_placed[PLAYER1] == 3 and self.pieces_placed[PLAYER2] == 3:
                self.phase = PHASE_MOVEMENT

        else:
            from_pos, to_pos = move
            self.board[from_pos] = EMPTY
            self.board[to_pos]   = current_player

        reward = 0
        if self._check_winner(current_player):
            self.done   = True
            self.winner = current_player
            reward      = 1

        self.turn = PLAYER2 if current_player == PLAYER1 else PLAYER1
        if not self.done and not self.get_legal_moves():
            self.done   = True
            self.winner = None
            reward      = 0

        return self.get_state(), reward, self.done, {"winner": self.winner}

    def _check_winner(self, player):
        """Return True if 'player' has all 3 pieces in a straight line."""
        for line in WINNING_LINES:
            if all(self.board[pos] == player for pos in line):
                return True
        return False

    def render(self):
        """Print an ASCII representation of the board matching the real layout."""
        s = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        b = self.board
        print(f"\n      {s[b[0]]}")
        print(f"     /|\\")
        print(f"    / | \\")
        print(f"   {s[b[1]]}--{s[b[2]]}--{s[b[3]]}")
        print(f"  /   |   \\")
        print(f" {s[b[4]]}----{s[b[5]]}----{s[b[6]]}")
        print(f"\nPhase: {self.phase} | Turn: Player {self.turn}")
        print(f"Legal moves: {self.get_legal_moves()}\n")