import cv2
import numpy as np
import subprocess
import os
import json
import chess
import chess.engine
import shutil
import sys

PIECES = {
    "KING": {
        "name": "King",
        "color_name": "Yellow",
        "hsv_lower": np.array([25, 150, 150]),   
        "hsv_upper": np.array([35, 255, 255]),
        "notation": "K"  
    },
    "QUEEN": {
        "name": "Queen",
        "color_name": "Magenta/Hot Pink",
        "hsv_lower": np.array([140, 100, 100]),
        "hsv_upper": np.array([170, 255, 255]),
        "notation": "Q"
    },
    "BISHOP": {
        "name": "Bishop",
        "color_name": "Cyan/Blue",
        "hsv_lower": np.array([80, 100, 100]),
        "hsv_upper": np.array([130, 255, 255]),
        "notation": "B"
    },
    "KNIGHT": {
        "name": "Knight",
        "color_name": "Bright Red",
        "hsv_lower": np.array([0, 100, 100]),
        "hsv_upper": np.array([10, 255, 255]),
        "notation": "N"  # 'N' for Knight in chess notation
    },
    "ROOK": {
        "name": "Rook",
        "color_name": "Orange",
        "hsv_lower": np.array([10, 100, 100]),
        "hsv_upper": np.array([20, 255, 255]),   # 28 MAX � DO NOT CROSS 30
        "notation": "R"
    },
    "PAWN": {
        "name": "Pawn",
        "color_name": "Green",
        "hsv_lower": np.array([35, 100, 100]),
        "hsv_upper": np.array([85, 255, 255]),
        "notation": "P"
    },
}


# Global variable to store previous positions for tracking
last_known_positions = {}

def cleanup_previous_runs(keep_latest=True):
    """Clean up images from previous runs, optionally keeping the most recent run"""
    print("Cleaning up old images...")
    
    # Define directories and files to clean
    debug_dir = "debug_squares"
    
    # List of files in the main directory to manage
    image_files = [
        "chess_capture.jpg", 
        "chessboard_detected.jpg",
        "threshold_image.jpg", 
        "failed_chessboard.jpg",
        "rotated_raw.jpg",
        "warped_board.jpg",
        "detected_pieces.jpg"
    ]
     # If we need to keep the latest run, first archive those files
    if keep_latest and os.path.exists("warped_board.jpg"):
        # Create archive directory if it doesn't exist
        if not os.path.exists("latest_run"):
            os.makedirs("latest_run")
        
        # Move all existing images to the archive
        for filename in image_files:
            if os.path.exists(filename):
                # Copy to archive
                shutil.copy2(filename, os.path.join("latest_run", filename))
    
    # Delete all existing images in the main directory
    for filename in image_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted {filename}")
    
    # Handle the debug squares directory
    if os.path.exists(debug_dir):
        # If we want to keep latest run, archive the debug directory
        if keep_latest:
            # Create archive debug directory
            latest_debug_dir = os.path.join("latest_run", debug_dir)
            if os.path.exists(latest_debug_dir):
                shutil.rmtree(latest_debug_dir)
            
            # Only copy if there's something to copy
            if os.path.exists(debug_dir):
                shutil.copytree(debug_dir, latest_debug_dir)
        
        # Remove the current debug directory
        shutil.rmtree(debug_dir)
        print(f"Cleaned up {debug_dir} directory")
    
    print("Cleanup complete")

def capture_image():
    """Capture a new image using libcamera"""
    try:
        command = ["libcamera-jpeg", "-o", "chess_capture.jpg", "-t", "2000"]
        subprocess.run(command, check=True)
        print("Image captured successfully")
        return cv2.imread("chess_capture.jpg")
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def find_and_warp_chessboard(img):
    """Find the chessboard in the image and warp it to a top-down view"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
     # Find the internal corners (7x7 grid points for an 8x8 chessboard)
    found, corners = cv2.findChessboardCorners(gray, (7, 7))
    if not found:
        print("Chessboard not found!")
        # Try with adaptive thresholding to improve pattern recognition
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite("threshold_image.jpg", thresh)
        found, corners = cv2.findChessboardCorners(thresh, (7, 7))
        if not found:
            cv2.imwrite("failed_chessboard.jpg", img)
            return None
            
    # Draw the detected internal corners
    detected_img = img.copy()
    cv2.drawChessboardCorners(detected_img, (7, 7), corners, found)
    cv2.imwrite("chessboard_detected.jpg", detected_img)
    
    # Create the full 8x8 grid corners by extrapolation
    corners = corners.reshape(-1, 2)
    grid_corners = np.zeros((9, 9, 2), dtype=np.float32)
    
    # Fill in the known interior corners
    for i in range(7):
        for j in range(7):
            grid_corners[i+1, j+1] = corners[i*7 + j]
    
    # Extrapolate outer edges
    # Top edge
    for j in range(1, 8):
        vector = grid_corners[1, j] - grid_corners[2, j]
        grid_corners[0, j] = grid_corners[1, j] + vector
    # Bottom edge
    for j in range(1, 8):
        vector = grid_corners[7, j] - grid_corners[6, j]
        grid_corners[8, j] = grid_corners[7, j] + vector
    
    # Left edge
    for i in range(1, 8):
        vector = grid_corners[i, 1] - grid_corners[i, 2]
        grid_corners[i, 0] = grid_corners[i, 1] + vector
    
    # Right edge
    for i in range(1, 8):
        vector = grid_corners[i, 7] - grid_corners[i, 6]
        grid_corners[i, 8] = grid_corners[i, 7] + vector
      # Corner points
    # Top-left
    vector_from_y = grid_corners[1, 0] - grid_corners[2, 0]
    vector_from_x = grid_corners[0, 1] - grid_corners[0, 2]
    grid_corners[0, 0] = grid_corners[1, 1] + vector_from_y + vector_from_x
     # Top-right
    vector_from_y = grid_corners[1, 8] - grid_corners[2, 8]
    vector_from_x = grid_corners[0, 7] - grid_corners[0, 6]
    grid_corners[0, 8] = grid_corners[1, 7] + vector_from_y + vector_from_x
    
    # Bottom-left
    vector_from_y = grid_corners[7, 0] - grid_corners[6, 0]
    vector_from_x = grid_corners[8, 1] - grid_corners[8, 2]
    grid_corners[8, 0] = grid_corners[7, 1] + vector_from_y + vector_from_x
     
    # Bottom-right
    vector_from_y = grid_corners[7, 8] - grid_corners[6, 8]
    vector_from_x = grid_corners[8, 7] - grid_corners[8, 6]
    grid_corners[8, 8] = grid_corners[7, 7] + vector_from_y + vector_from_x
     # Create a warped perspective view of the chessboard
    warp_size = (800, 800)
    
    # Define the destination points for the perspective transform
    dst_points = np.array([ 
        [0, 0], 
        [warp_size[0], 0], 
        [warp_size[0], warp_size[1]], 
        [0, warp_size[1]] 
    ], dtype=np.float32)
    
    # Use the four corner points of the chessboard
    src_points = np.array([ 
        grid_corners[0, 0],  # Top-left 
        grid_corners[0, 8],  # Top-right 
        grid_corners[8, 8],  # Bottom-right 
        grid_corners[8, 0]   # Bottom-left 
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transformation
    warped = cv2.warpPerspective(img, M, warp_size)
    
    return warped

def enhance_image_for_color(img):
    """Apply multiple image processing techniques to enhance color detection"""
    # Make a copy to avoid modifying the original
    enhanced = img.copy()
    
    # Method 1: Convert to HSV and increase saturation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.7, 0, 255).astype(np.uint8)  # Increase saturation more
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
     # Method 2: Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Method 3: Apply CLAHE for better contrast on each channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Method 4: Slightly sharpen the image
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9.5, -1], 
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def detect_pieces(warped_img):
    """Detect chess pieces based on their colored markings and assign teams"""
    # Get image dimensions
    height, width = warped_img.shape[:2]
    square_size = width // 8
    
    # Convert to HSV color space for better color detection
    hsv_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    
    # Create output visualization
    result_img = warped_img.copy()
    
    # Dictionary to store detected pieces
    piece_positions = {}
    
    # Create debug directory
    debug_dir = "debug_squares"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Process each square
    for row in range(8):
        for col in range(8):
            # Get square coordinates
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
              
            # Chess position in algebraic notation
            position = f"{chr(97 + col)}{8 - row}"  # e.g., "a1", "h8"
            
            # Extract square image
            square_img = warped_img[y1:y2, x1:x2]
            square_hsv = hsv_img[y1:y2, x1:x2]
              # Save square for debugging
            cv2.imwrite(f"{debug_dir}/square_{position}.jpg", square_img)
            
            # Apply multiple methods to enhance color detection
            enhanced_img = enhance_image_for_color(square_img)
            enhanced_hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            cv2.imwrite(f"{debug_dir}/enhanced_{position}.jpg", enhanced_img)
            
            # Check for pieces
            piece_detected = False
            
            for piece_type, piece_info in PIECES.items():
                # Try multiple processing techniques to improve detection
                masks = []
                
                # 1. Standard HSV range detection
                mask1 = cv2.inRange(square_hsv, piece_info["hsv_lower"], piece_info["hsv_upper"])
                masks.append(mask1)
                
                # 2. Enhanced image detection
                mask2 = cv2.inRange(enhanced_hsv, piece_info["hsv_lower"], piece_info["hsv_upper"])
                masks.append(mask2)
                
                # 3. Apply histogram equalization and try again
                eq_hsv = square_hsv.copy()
                eq_hsv[:,:,2] = cv2.equalizeHist(eq_hsv[:,:,2])
                mask3 = cv2.inRange(eq_hsv, piece_info["hsv_lower"], piece_info["hsv_upper"])
                masks.append(mask3)
                
                # 4. Try with a slightly expanded HSV range for better coverage
                extended_lower = piece_info["hsv_lower"].copy()
                extended_upper = piece_info["hsv_upper"].copy()
                # Expand the range slightly (5 units in each direction)
                extended_lower[0] = max(0, extended_lower[0] - 5)
                extended_lower[1] = max(0, extended_lower[1] - 5)
                extended_lower[2] = max(0, extended_lower[2] - 5)
                extended_upper[0] = min(180, extended_upper[0] + 5)
                extended_upper[1] = min(255, extended_upper[1] + 5)
                extended_upper[2] = min(255, extended_upper[2] + 5)
                mask4 = cv2.inRange(enhanced_hsv, extended_lower, extended_upper)
                masks.append(mask4)
                
                # Combine masks
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                
                # Clean up the mask
                kernel = np.ones((3,3), np.uint8)
                refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
                
                # Look for meaningful colored regions
                contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 # Filter small noise and look for significant areas
                # Reduce the minimum contour area for more sensitivity
                meaningful_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 15]
                
                # Only proceed if there are meaningful contours
                if meaningful_contours:
                    # Calculate total area of meaningful colored regions
                    total_area = sum(cv2.contourArea(cnt) for cnt in meaningful_contours)
                    
                    # Set a threshold based on the piece type and expected marker size
                    # Lower threshold for better detection
                    min_area_threshold = square_size * square_size * 0.015  # 1.5% of square
                    
                    if total_area > min_area_threshold:
                       
                        rank = int(position[1])
                        if rank <= 2:
                            team = "WHITE"
                        elif rank >= 7:
                            team = "BLACK"
                        else:
                            team = "UNKNOWN"  
                        
                        print(f"Detected {team} {piece_info['name']} at {position}")
                        
                        piece_positions[position] = {
                            "type": piece_type,
                            "notation": piece_info["notation"],
                            "team": team,
                            "position": position
                        }
                        
                        # Draw contours
                        cv2.drawContours(result_img, meaningful_contours, -1, (0, 255, 255), 1, offset=(x1, y1))
                        
                        # Save debug mask
                        cv2.imwrite(f"{debug_dir}/mask_{position}_{piece_type}.jpg", refined_mask)
                        
                        piece_detected = True
                        break
    
    # Save the result image with detections
    cv2.imwrite("detected_pieces.jpg", result_img)
    
    return result_img, piece_positions

def track_and_update_teams(new_positions):
    """Track pieces across moves and maintain team information"""
    global last_known_positions
    
    # Print debug info
    print(f"Last known positions: {len(last_known_positions)} pieces")
    print(f"New positions: {len(new_positions)} pieces")
    
  
    for position, piece_info in new_positions.items():
        if position in last_known_positions:
            old_team = last_known_positions[position]["team"]
            if old_team != "UNKNOWN":
                new_positions[position]["team"] = old_team
                print(f"Preserved team for {piece_info['type']} at {position}: {old_team}")

    missing_pieces = {}
    for old_pos, old_info in last_known_positions.items():
        if old_pos not in new_positions:
            piece_type = old_info["type"]
            team = old_info["team"]
          
            if team != "UNKNOWN":
                
                if piece_type not in missing_pieces:
                    missing_pieces[piece_type] = []
                missing_pieces[piece_type].append({"pos": old_pos, "team": team})
                print(f"Missing {team} {piece_type} from {old_pos}")
    
    # Now update any remaining UNKNOWN teams in new positions
    for position, piece_info in new_positions.items():
        if piece_info["team"] == "UNKNOWN":
            piece_type = piece_info["type"]
            
            # Check if this type of piece was previously detected and is now missing
            if piece_type in missing_pieces and missing_pieces[piece_type]:
                # Get the first missing piece of this type
                missing_piece = missing_pieces[piece_type].pop(0)
                
                # Update team information
                new_positions[position]["team"] = missing_piece["team"]
                print(f"Tracking: {piece_type} likely moved from {missing_piece['pos']} to {position}, team is {missing_piece['team']}")
    
    # Update last known positions for next time
    last_known_positions = new_positions.copy()
    
    # Update visualization
    result_img = cv2.imread("warped_board.jpg")
    if result_img is not None:
        height, width = result_img.shape[:2]
        square_size = width // 8
        
        for position, piece_info in new_positions.items():
            # Get square coordinates
            col = ord(position[0]) - ord('a')
            row = 8 - int(position[1])
            
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            team = piece_info["team"]
            team_color = (0, 0, 255) if team == "BLACK" else (255, 255, 255) if team == "WHITE" else (0, 255, 255)
            
            # Draw rectangle with team color
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with piece name
            label = f"{team[0]}_{piece_info['notation']}"
            cv2.putText(result_img, label, (x1 + 5, y1 + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
        
        # Save the updated image
        cv2.imwrite("detected_pieces.jpg", result_img)
    
    return new_positions

def generate_board_state(piece_positions):
    """Generate a complete representation of the chess board state"""
    # Create an 8x8 empty board
    board = [['.' for _ in range(8)] for _ in range(8)]
    
    # Dictionary for piece notation
    notation_map = {
        "WHITE": {
            "KING": "K", "QUEEN": "Q", "BISHOP": "B", "KNIGHT": "N", "ROOK": "R", "PAWN": "P"
        },
        "BLACK": {
            "KING": "k", "QUEEN": "q", "BISHOP": "b", "KNIGHT": "n", "ROOK": "r", "PAWN": "p"
        },
        "UNKNOWN": {
            "KING": "?", "QUEEN": "?", "BISHOP": "?", "KNIGHT": "?", "ROOK": "?", "PAWN": "?"
        }
    }
    
    # Fill in detected pieces
    for position, piece_info in piece_positions.items():
        col = ord(position[0]) - ord('a')
        row = 8 - int(position[1])
        
        team = piece_info["team"]
        piece_type = piece_info["type"]
        
        # Use uppercase for white, lowercase for black, ? for unknown
        notation = notation_map[team][piece_type]
        
        board[row][col] = notation
    
    # Create string representation
    board_str = ""
    for row in board:
        board_str += ' '.join(row) + "\n"
    
    # Also create FEN notation (Forsyth-Edwards Notation)
    fen = generate_fen(board)
    
    return {
        "board_array": board,
        "board_string": board_str,
        "fen": fen
    }

def generate_fen(board):
    """Generate FEN notation from the board array"""
    fen = ""
    
    # Board position
    for row in board:
        empty_count = 0
        for cell in row:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                # Make sure no invalid characters are added to the FEN
                if cell in "KQBNRPkqbnrp":
                    fen += cell
                else:
                    # Replace any unknown characters with placeholders based on position
                    # Default to empty space (increase empty count) to avoid invalid FEN
                    empty_count += 1
        
        if empty_count > 0:
            fen += str(empty_count)
        
        # Add row separator
        fen += "/"
    
    # Remove trailing slash
    fen = fen[:-1]
    
    # Validate FEN before returning (make sure no invalid chars remain)
    valid_chars = "12345678KQBNRPkqbnrp/"
    for char in fen:
        if char not in valid_chars:
            print(f"Warning: Invalid character '{char}' found in FEN, replacing with '1'")
            fen = fen.replace(char, '1')
    
    # Check for any consecutive slash issues (//)
    while "//" in fen:
        fen = fen.replace("//", "/8/")
    
    # Add placeholder for other FEN components (active color, castling, etc.)
    fen += " w KQkq - 0 1"
    
    return fen

def get_chess_advice(fen, for_team="WHITE"):
    """Get chess advice for the current position using Stockfish"""
    try:
        # Create a board from the FEN
        board = chess.Board(fen)
        
        # Determine whose turn it is
        is_white_turn = board.turn == chess.WHITE
        team_to_move = "WHITE" if is_white_turn else "BLACK"
        
        print(f"Turn: {team_to_move}")
        
        # Only provide advice if it's the requested team's turn
        if team_to_move != for_team:
            return f"It's {team_to_move}'s turn. Waiting for their move."
        
        # Try multiple possible locations for Stockfish
        stockfish_paths = [
            "/usr/bin/stockfish",
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "stockfish"  # This will work if stockfish is in PATH
        ]
        
        engine = None
        for path in stockfish_paths:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(path)
                print(f"Found Stockfish at {path}")
                break
            except Exception:
                continue
        
        if engine is None:
            return "Stockfish engine not found. Please install Stockfish chess engine."
        
        # Set a reasonable thinking time
        time_limit = chess.engine.Limit(time=0.5)
        
        # Get the best move and score
        result = engine.analyse(board, time_limit, multipv=3)
        
        # Parse the results
        advice = []
        print("\nChess Engine Analysis:")
        print(f"Position evaluation: {result[0]['score'].white()}")
        for i, entry in enumerate(result):
            move = entry["pv"][0]
            score = entry["score"].white()
            san_move = board.san(move)
            
            # Generate a human-friendly explanation
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            piece_name = get_piece_name(board.piece_at(move.from_square))
            
            if i == 0:
                explanation = f"Best move: Move your {piece_name} from {from_square} to {to_square} ({san_move})."
            else:
                explanation = f"Alternative {i}: Move your {piece_name} from {from_square} to {to_square} ({san_move})."
            
            advice.append(explanation)
        
        # Add some general advice based on the score
        if score.score() is not None:
            score_value = score.score() / 100.0  # Convert to pawn units
            if score_value > 3:
                advice.append("You have a winning advantage. Look for tactics to convert your advantage.")
            elif score_value > 1.5:
                advice.append("You have a clear advantage. Consider simplifying to an endgame.")
            elif score_value > 0.5:
                advice.append("You have a slight advantage. Control the center and develop your pieces.")
            elif score_value > -0.5:
                advice.append("The position is roughly equal. Focus on piece development and control of key squares.")
            elif score_value > -1.5:
                advice.append("You're at a slight disadvantage. Look for counterplay and avoid exchanges.")
            else:
                advice.append("You're in a difficult position. Look for defensive resources and tactical chances.")
        
        # Quit the engine
        engine.quit()
        
        # Save the advice to a file for the UI to read
        chess_advice = "\n".join(advice)
        with open("chess_advice.txt", "w") as f:
            f.write(chess_advice)
        
        return chess_advice
    except Exception as e:
        print(f"Error generating chess advice: {e}")
        error_message = f"Couldn't analyze the position. Error: {str(e)}"
        
        # Save the error message to the advice file
        with open("chess_advice.txt", "w") as f:
            f.write(error_message)
            
        return error_message

def get_piece_name(piece):
    """Get a friendly name for a chess piece"""
    if piece is None:
        return "piece"
    
    piece_names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king"
    }
    
    return piece_names.get(piece.piece_type, "piece")

def reset_game():
    """Reset the chess detection system for a new game"""
    print("Resetting for a new game...")
    
    # Files to delete
    files_to_delete = [
        "piece_history.json",
        "board_state.json",
        "chess_advice.txt"
    ]
    
    # Delete all history files
    for filename in files_to_delete:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted {filename}")
    
    # Reset global variable
    global last_known_positions
    last_known_positions = {}
    
    print("Game reset complete. Ready for a new game!")

def main():
    """Main function to detect chess pieces and generate board state"""
    # Check if we should reset the game
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_game()
        return
    
    print("Chess Piece Detector Starting...")
    
    global last_known_positions
    
    # Clean up files from previous runs before starting
    # Commented out to keep old files as requested
    # cleanup_previous_runs(keep_latest=True)
    
    # Load previous positions if available
    if os.path.exists("piece_history.json"):
        try:
            with open("piece_history.json", "r") as f:
                last_known_positions = json.load(f)
            print(f"Loaded history with {len(last_known_positions)} pieces")
        except Exception as e:
            print(f"Error loading piece history: {e}")
    
    # Capture or load an image
    img = capture_image()
    if img is None:
        print("Using sample image if available...")
        img = cv2.imread("chess_sample.jpg")
        if img is None:
            print("No image available to process")
            return
     # Rotate the image to match your viewing perspective
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Adjust as needed
    
    # Save the rotated raw image for verification
    cv2.imwrite("rotated_raw.jpg", img)
    
    # Find and warp the chessboard
    print("Finding chessboard...")
    warped_img = find_and_warp_chessboard(img)
    if warped_img is None:
        print("Failed to find chessboard")
        return
    print("Chessboard found and warped")
    
    # Save warped image
    cv2.imwrite("warped_board.jpg", warped_img)
    
    # Detect pieces
    print("Detecting pieces...")
    result_img, piece_positions = detect_pieces(warped_img)
    
    # Track pieces and update team information
    print("Updating team information based on tracking...")
    piece_positions = track_and_update_teams(piece_positions)
    
    # Save updated positions for next time
    with open("piece_history.json", "w") as f:
        json.dump(piece_positions, f, indent=2)
    
    # After generating the board state
    board_state = generate_board_state(piece_positions)
    print("\nCurrent Board State:")
    print(board_state["board_string"])
    print("\nFEN Notation:")
    print(board_state["fen"])

    # Get chess advice for white
    print("\nGetting chess advice...")
    advice = get_chess_advice(board_state["fen"], for_team="WHITE")
    print("\nChess Advice for White:")
    print(advice)
    
    # Save board state to JSON
    with open("board_state.json", "w") as f:
        json.dump(board_state, f, indent=2)
    print("Results saved")
    
    # Display detected pieces with team information
    for position, piece_info in piece_positions.items():
        print(f"{piece_info['team']} {piece_info['type']} at {position}")
        
if __name__ == "__main__":
    main()