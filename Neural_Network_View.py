import pygame
from neural_network_a import Neural_Network

pygame.init()

window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Neural Network Training")

clock = pygame.time.Clock()

def draw_text(surface, text, position, font_size=20, color=(255, 255, 255)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

#draw display
def draw_display():
    h = 15
    w = 50

    draw_text(screen, "Weights:", (10, 10))
    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (10 + j * w , 30 + i * h))
    c = j+2

    draw_text(screen, "Biases:", (10 + (c*w), 10))
    for i, row in enumerate(biases):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (10 + (c*w) + j * w , 30 + i * h))


#NEURAL NETWORK




nn = Neural_Network()
weights, biases = nn.network[0].print()

#print(weights, biases)















#Screen Display
while True:
    clock.tick(10)
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    screen.fill((0, 0, 0)) 
    draw_display()

    pygame.display.flip()