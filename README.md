# ♟️ ChessGPT

A Raspberry Pi-based computer vision system that detects physical chess board positions using a camera and provides AI-powered move suggestions via the Stockfish chess engine.

---

## 📌 Overview

Chess Vision AI bridges the physical and digital worlds of chess. Point a camera at a real chessboard, and the system will:
- Detect the board and identify piece positions using OpenCV
- Track which pieces belong to which player
- Analyze the current position using Stockfish
- Display move suggestions and strategic advice in a GUI panel

This project was built as a semester-long project using a Raspberry Pi, a camera module, and Python.

---

## 🎥 Demo

---<img width="1771" height="750" alt="chess32" src="https://github.com/user-attachments/assets/d5ad14cc-cacb-4c81-a468-f94f725a8d0e" />
---<img width="1771" height="750" alt="IMG_4303" src="https://github.com/user-attachments/assets/a2931f79-93d0-463c-8259-4cbd875afbab" />


## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenCV | Camera input & image processing |
| Stockfish | Chess engine for move analysis |
| python-chess | Board state management & engine interface |
| Tkinter | Desktop GUI |
| Raspberry Pi | Hardware platform |

---

## 📁 Project Structure

```
chess-vision-ai/
├── main.py                  # Main application entry point & GUI
├── src/
│   └── color_detector.py    # Chess piece detection logic
├── data/
│   └── chess_images/        # Reference images for piece recognition
├── debug_squares/           # Debug output images of detected squares
├── latest_run/              # Output from the most recent detection run
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Raspberry Pi (any model with camera support)
- Pi Camera Module or USB webcam
- Python 3.x
- Stockfish chess engine

### 1. Clone the repository
```bash
git clone https://github.com/achev2/chess-vision-ai.git
cd chess-vision-ai
```

### 2. Install Stockfish
```bash
sudo apt-get install stockfish
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare your chess set
- Apply colored stickers or markers to your chess pieces
- Use consistent colors per team so the color detector can distinguish sides
- Ensure good lighting when running the system

---

## ▶️ Running the App

```bash
python main.py
```

### In the GUI:
- Click **"Detect Board"** to scan the current board position via camera
- The AI Output panel on the right will display Stockfish's suggested moves
- Click **"New Game"** to reset the board to the starting position

---

## 🧠 How It Works

1. **Image Capture** — OpenCV captures a frame from the connected camera
2. **Board Detection** — The system identifies the 8x8 grid using contour and line detection
3. **Piece Recognition** — Each square is analyzed for colored markers to determine piece type and team
4. **State Tracking** — Piece positions are stored and updated as moves are made
5. **Engine Analysis** — The board state is passed to Stockfish via python-chess for evaluation
6. **Output** — Suggested moves and strategic notes are displayed in the GUI

---

## 📦 Generated Files (at runtime)

These files are created when the app runs and are excluded from the repo via `.gitignore`:

- `board_state.json` — Current board position
- `piece_history.json` — History of piece movements
- `chess_advice.txt` — Latest AI move suggestions

---

## 🔧 Troubleshooting

**Board not detected correctly?**
- Make sure the full board is visible and well-lit
- Ensure there's enough contrast between light and dark squares

**Pieces not recognized?**
- Check that colored markers are bright and clearly visible
- Verify HSV color ranges in `color_detector.py` match your markers

**Stockfish not found?**
- Confirm Stockfish is installed: `which stockfish`
- If installed in a custom path, update `stockfish_paths` in `color_detector.py`

---

## 👤 Author

**achev2**  
Senior Computer Science Student  
[GitHub](https://github.com/achev2)

---

## 📄 License

This project is open source and available under the MIT License.
