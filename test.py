import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# --- 1. SHAPE HELPERS ---
def draw_sphere(radius=1, color=(1, 0, 0)):
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color + (1.0,))
    sphere = gluNewQuadric()
    gluQuadricNormals(sphere, GLU_SMOOTH)
    gluSphere(sphere, radius, 16, 16)

def draw_cylinder(radius=0.5, height=1, color=(0, 0, 1)):
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color + (1.0,))
    cyl = gluNewQuadric()
    gluQuadricNormals(cyl, GLU_SMOOTH)
    gluCylinder(cyl, radius, radius, height, 16, 1)

# --- 2. THE CHARACTER (Woody-like) ---
def draw_woody(walk_cycle):
    # walk_cycle: -1 to 1

    # --- TORSO (Parent) ---
    glPushMatrix()
    glTranslatef(0, 1.0, 0) # Move up from ground
    
    # Shirt (Yellow)
    glPushMatrix()
    glScalef(0.8, 1.2, 0.6)
    draw_sphere(1.0, color=(1.0, 0.8, 0.2)) # Yellow-ish shirt
    glPopMatrix()

    # Vest (Brown)
    glPushMatrix()
    glTranslatef(0, 0.1, 0.05)
    glScalef(0.85, 1.1, 0.65)
    draw_sphere(1.0, color=(0.4, 0.2, 0.1)) # Brown vest
    glPopMatrix()
    
    # --- HEAD & HAT ---
    glPushMatrix()
    glTranslatef(0, 1.3, 0)
    # Head (Skin tone)
    draw_sphere(0.6, color=(0.8, 0.6, 0.5))

    # Hat (Brown)
    glPushMatrix()
    glTranslatef(0, 0.5, 0)
    glRotatef(-10, 1, 0, 0) # Tilt hat
    # Brim
    glPushMatrix()
    glScalef(1.5, 0.2, 1.5)
    draw_sphere(0.7, color=(0.35, 0.15, 0.05))
    glPopMatrix()
    # Crown
    glTranslatef(0, 0.3, 0)
    draw_cylinder(0.5, 0.6, color=(0.35, 0.15, 0.05))
    glTranslatef(0, 0, 0.6)
    draw_sphere(0.5, color=(0.35, 0.15, 0.05))
    glPopMatrix()
    glPopMatrix() # End Head

    # --- ARMS ---
    for side in [-1, 1]: # -1 for left, 1 for right
        glPushMatrix()
        glTranslatef(side * 0.8, 0.8, 0)
        glRotatef(side * walk_cycle * 30, 1, 0, 0) # Swing arms
        glRotatef(90, 1, 0, 0) # Point down
        # Sleeve
        draw_cylinder(0.2, 0.6, color=(1.0, 0.8, 0.2))
        glTranslatef(0, 0, 0.6)
        # Forearm/Hand (Skin tone)
        draw_cylinder(0.2, 0.6, color=(0.8, 0.6, 0.5))
        glTranslatef(0, 0, 0.6)
        draw_sphere(0.25, color=(0.8, 0.6, 0.5))
        glPopMatrix()

    # --- LEGS ---
    for side in [-1, 1]:
        glPushMatrix()
        glTranslatef(side * 0.4, -0.8, 0)
        glRotatef(-side * walk_cycle * 30, 1, 0, 0) # Swing legs opposite to arms
        glRotatef(90, 1, 0, 0)
        # Jean Leg
        draw_cylinder(0.25, 1.2, color=(0.2, 0.3, 0.6)) # Blue jeans
        glTranslatef(0, 0, 1.2)
        # Boot (Brown)
        draw_cylinder(0.3, 0.5, color=(0.35, 0.15, 0.05))
        glTranslatef(0, 0.15, 0.5)
        glScalef(1.0, 1.5, 1.0)
        draw_sphere(0.3, color=(0.35, 0.15, 0.05))
        glPopMatrix()

    glPopMatrix() # End Torso/Body

# --- 3. MAIN LOOP ---
def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, -1.0, -10) # Move camera back and down slightly

    # Basic Lighting
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_DEPTH_TEST); glEnable(GL_NORMALIZE)
    glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 10, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 0.9, 0.8, 1)) # Slightly warm light

    clock = pygame.time.Clock()
    time = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); quit()

        time += 0.1
        walk_cycle = math.sin(time * 1.5) # Speed of walking

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        glRotatef(15, 0, 1, 0) # Slight side view
        draw_woody(walk_cycle)
        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()