import pygame

# Initialize Pygame
pygame.init()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Screen dimensions
width, height = 800, 800

# Create the screen
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Chess Game')

# Load images
pieces = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR',
          'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp',
          'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp',
          'wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']

# Load piece images
piece_images = {}
for piece in pieces:
    piece_images[piece] = pygame.image.load(f'images/{piece}.png')

# Draw the board
def draw_board():
    colors = [white, black]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col*100, row*100, 100, 100))

# Draw the pieces
def draw_pieces(board):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '--':
                screen.blit(piece_images[piece], pygame.Rect(col*100, row*100, 100, 100))

# Main game loop
def main():
    board = [
        ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_board()
        draw_pieces(board)
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
