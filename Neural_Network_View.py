import pygame
from neural_network_a import Neural_Network
import numpy as np

pygame.init()

window_size = (1800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Neural Network Training")

clock = pygame.time.Clock()

def draw_text(surface, text, position, font_size=20, color=(255, 255, 255)):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

#draw display
def draw_display(e, epochs, weights, biases, weights2, biases2, output, output1, output2, output3, output4, error, grad, grad1, grad2, grad3, grad4):
    screen.fill((0, 0, 0)) 
    x = 10      #cell x cord
    y = 50      #cell y cord
    h = 15      #cell height
    w = 50      #cell width
    c = 0       #cell count
    j = 0

    draw_text(screen, f"Epochs: {epochs} / {e}", (x, y-40))

    draw_text(screen, "Input:", (x, y))
    for i, row in enumerate(output):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + j * w , y +20 + i * h))
    c = c+j+2


    draw_text(screen, "Weights_L1:", (x + (c*w), y))
    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Biases_L1:", (x + (c*w), y))
    for i, row in enumerate(biases):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Out_L1:", (x + (c*w), y))
    for i, row in enumerate(output1):
        for j, value in enumerate(row):
            value = value[0] if isinstance(value, np.ndarray) else value
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Out_A1:", (x + (c*w), y))
    for i, row in enumerate(output2):
        for j, value in enumerate(row):
            value = value[0] if isinstance(value, np.ndarray) else value
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Weights_L2:", (x + (c*w), y))
    for i, row in enumerate(weights2):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Biases_L2:", (x + (c*w), y))
    for i, row in enumerate(biases2):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Out_L2:", (x + (c*w), y))
    for i, row in enumerate(output3):
        for j, value in enumerate(row):
            value = value[0] if isinstance(value, np.ndarray) else value
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2
    
    draw_text(screen, "Out_A2:", (x + (c*w), y))
    for i, row in enumerate(output4):
        for j, value in enumerate(row):
            value = value[0] if isinstance(value, np.ndarray) else value
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Grad_A2in:", (x + (c*w), y))
    for i, row in enumerate(grad):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2
    
    draw_text(screen, "Grad_A2out:", (x + (c*w), y))
    for i, row in enumerate(grad1):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Grad_L2out:", (x + (c*w), y))
    for i, row in enumerate(grad2):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Grad_A1out:", (x + (c*w), y))
    for i, row in enumerate(grad3):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2

    draw_text(screen, "Grad_L1out:", (x + (c*w), y))
    for i, row in enumerate(grad4):
        for j, value in enumerate(row):
            draw_text(screen, f"{value:.4f}", (x + (c*w) + j * w , y +20 + i * h))
    c = c+j+2
  
    draw_text(screen, f"Error: {error:.10f}", (x + (c*w), y))





    pygame.display.flip()


#NEURAL NETWORK

nn = Neural_Network()

epochs = 10000
epc = 0


#Screen Display
while True:
    clock.tick(1000)
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


    #weights, biases = nn.network[0].print()
    weights, biases = nn.dense1.print()
    weights2, biases2 = nn.dense2.print()
    output, output1, output2, output3, output4, error, grad, grad1, grad2, grad3, grad4 = nn.print()
  
    if epc < epochs:
        nn.train()
        epc += 1

    draw_display(epc, epochs, weights, biases, weights2, biases2, output, output1, output2, output3, output4, error, grad, grad1, grad2, grad3, grad4)

    