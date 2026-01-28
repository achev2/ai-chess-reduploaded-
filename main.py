import tkinter as tk
import os
import subprocess
import json
import time

class ChessGPT:
    def __init__(self, master):
        self.master = master
        self.master.title("ChessGPT")
        self.master.minsize(1000, 650)
        self.master.configure(bg="#2c3e50")
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=3)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=2)

        self.selected_piece = None
        self.board_state = {}

        self.load_piece_images()
        self.setup_ui_components()
        self.setup_pieces()

    def load_piece_images(self):
        base_path = "/home/chessgpt/project/aichess/data/chess_images"
        self.piece_images = {}

        piece_files = {
            'K': "white-king.png",
            'Q': "white-queen.png",
            'R': "white-rook.png",
            'B': "white-bishop.png",
            'N': "white-knight.png",
            'P': "white-pawn.png",
            'k': "black-king.png",
            'q': "black-queen.png",
            'r': "black-rook.png",
            'b': "black-bishop.png",
            'n': "black-knight.png",
            'p': "black-pawn.png"
        }

        for key, filename in piece_files.items():
            image_path = os.path.join(base_path, filename)
            img = tk.PhotoImage(file=image_path)
            img = img.subsample(2, 2)  # Resize smaller
            self.piece_images[key] = img

    def setup_ui_components(self):
        self.title_frame = tk.Frame(self.master, bg="#1e1e1e", height=80)
        self.title_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.title_frame.grid_propagate(False)

        self.title_label = tk.Label(
            self.title_frame,
            text="ChessGPT",
            font=("Helvetica", 28, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")

        self.board_container = tk.Frame(self.master, bg="#2c3e50", padx=20, pady=20)
        self.board_container.grid(row=1, column=0, sticky="nsew")

        self.create_chessboard()

        self.button_panel = tk.Frame(self.master, bg="#2c3e50")
        self.button_panel.grid(row=1, column=1, sticky="n", pady=60)

        self.detect_button = tk.Button(
            self.button_panel,
            text="Detect Board",
            font=("Helvetica", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=12,
            relief="flat",
            highlightthickness=0,
            bd=0,
            command=self.detect_board
        )
        self.detect_button.pack()

        self.new_game_button = tk.Button(
            self.button_panel,
            text="New Game",
            font=("Helvetica", 12, "bold"),
            bg="#2ecc71",
            fg="white",
            padx=20,
            pady=12,
            relief="flat",
            highlightthickness=0,
            bd=0,
            command=self.new_game
        )
        self.new_game_button.pack(pady=20)

        self.ai_frame = tk.Frame(self.master, bg="white", width=300, height=500)
        self.ai_frame.grid(row=1, column=2, sticky="nsew", padx=20, pady=20)
        self.ai_frame.pack_propagate(False)

        self.ai_title = tk.Label(
            self.ai_frame,
            text="AI Strategy",
            font=("Helvetica", 16, "bold"),
            bg="white",
            fg="#2c3e50",
            pady=10
        )
        self.ai_title.pack(fill="x")

        self.ai_output = tk.Text(
            self.ai_frame,
            wrap=tk.WORD,
            font=("Helvetica", 12),
            bg="white",
            relief="flat",
            padx=15,
            pady=15,
            height=15
        )
        self.ai_output.pack(expand=True, fill="both", padx=10, pady=10)

        self.update_ai_output("Welcome to ChessGPT!\nClick 'Detect Board' to begin.")

    def create_chessboard(self):
        self.board_frame = tk.Frame(self.board_container, bg="#34495e", padx=10, pady=10)
        self.board_frame.pack(expand=True)
        self.square_size = 65
        self.squares = {}

        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        for i, file in enumerate(files):
            file_label = tk.Label(self.board_frame, text=file, font=("Helvetica", 10), bg="#34495e", fg="white")
            file_label.grid(row=0, column=i+1, sticky="s")

        for i in range(8):
            rank_label = tk.Label(self.board_frame, text=str(8-i), font=("Helvetica", 10), bg="#34495e", fg="white")
            rank_label.grid(row=i+1, column=0, sticky="e")

        for row in range(8):
            for col in range(8):
                color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
                square = tk.Frame(
                    self.board_frame,
                    width=self.square_size,
                    height=self.square_size,
                    bg=color
                )
                square.grid(row=row+1, column=col+1)
                square.grid_propagate(False)
                self.squares[(row, col)] = square

    def setup_pieces(self):
        pass  # Start empty until detection

    def place_piece(self, row, col, piece_type):
        square = self.squares[(row, col)]
        image = self.piece_images.get(piece_type)
        if image:
            piece_label = tk.Label(square, image=image, bg=square['bg'])
            piece_label.image = image
            piece_label.place(relx=0.5, rely=0.5, anchor="center")
            self.board_state[(row, col)] = {'type': piece_type, 'label': piece_label}

    def remove_piece(self, row, col):
        if (row, col) in self.board_state:
            self.board_state[(row, col)]['label'].destroy()
            del self.board_state[(row, col)]

    def detect_board(self):
        self.update_ai_output("Scanning board... please wait...")

        try:
            subprocess.run(["python3", "src/color_detector.py"], check=True)
            time.sleep(1)  # Wait a bit to ensure file writing
        except Exception as e:
            self.update_ai_output(f"Detection failed: {str(e)}")
            return

        # Load board
        if os.path.exists("board_state.json"):
            with open("board_state.json", "r") as f:
                board_data = json.load(f)
            self.update_board_from_detection(board_data)
        else:
            self.update_ai_output("Detection failed: no board_state.json found.")
            return

        # Load advice
        if os.path.exists("chess_advice.txt"):
            with open("chess_advice.txt", "r") as f:
                advice = f.read()
            self.update_ai_output(advice)
        else:
            self.update_ai_output("No advice available.")

    def update_board_from_detection(self, board_data):
        board_array = board_data.get("board_array", [])

        # Clear old pieces
        for pos in list(self.board_state.keys()):
            self.remove_piece(pos[0], pos[1])

        # Place new pieces
        for row_idx, row in enumerate(board_array):
            for col_idx, piece in enumerate(row):
                if piece != '.':
                    self.place_piece(row_idx, col_idx, piece)

    def new_game(self):
        self.update_ai_output("Starting a new game... Resetting board and files.")

        files_to_delete = ["piece_history.json", "board_state.json", "chess_advice.txt"]
        for filename in files_to_delete:
            if os.path.exists(filename):
                os.remove(filename)

        # Clear all pieces
        for pos in list(self.board_state.keys()):
            self.remove_piece(pos[0], pos[1])

    def update_ai_output(self, text):
        self.ai_output.config(state=tk.NORMAL)
        self.ai_output.delete(1.0, tk.END)
        self.ai_output.insert(tk.END, text)
        self.ai_output.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = ChessGPT(root)
    root.mainloop()

if __name__ == "__main__":
    main()