import Solver
import pygame as pg
from pygame.locals import *
import time
size = 70 						# size of a square in px
bg_white = (255,255,255)		# chess white square bg color
bg_black = (220,210,235)			# enter screen color
def ask(screen):
	background = pg.image.load('img/queen.jpeg')
	background = pg.transform.scale(background,(900,300))
	screen.fill(bg_black)
	screen.blit(background, (0, 0))
	img = pg.image.load('img/white.png')
	pg.display.set_icon(img)
	pg.display.update()
	clock = pg.time.Clock()
	font = pg.font.Font(None, 32)
	done = False
	inp = ""
	
	while not done:
		for event in pg.event.get():
			surf = font.render(" ENTER THE NUMBER OF QUEENS: "+inp, True, (254,254,254), (0,0,0))
			rect = surf.get_rect()
			rect.center = (450, 150)
			screen.blit(surf,rect)
			
			pg.display.update()
			if event.type == pg.QUIT:
				done = True
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_RETURN:
					done = True
				elif event.key == pg.K_BACKSPACE:
					inp = inp[:-1]
				else:
					inp+=event.unicode
	return inp
	clock.tick(30)

def drawSoln(screen, soln, posArray):
	img = [pg.transform.scale(pg.image.load('img/Black2.png'), (size,size))]
	img.append(pg.transform.scale(pg.image.load('img/White2.png'), (size,size)))
	col = 0
	print(soln)
	for i in soln:
		time.sleep(0.7)
		x,y = posArray[col][i]
		screen.blit(img[(i+col)%2], (x,y))
		col+=1
		pg.display.update()

def main():
	pg.init()
	screen = pg.display.set_mode((900,300))
	pg.display.set_caption('N Queens:')
	n=int(ask(screen))
	done = False
	screen.fill(bg_black)
	screen = pg.display.set_mode((size*n + 4, size*n + 4))
	pg.display.set_caption('Solution of N Queens')
	pg.display.update()
	
	posArray = []
	
	for x in range(n):
		tempPos = []
		for y in range(n):
			tempPos.append((size*(x), size*(y)))
			if((x+y)&1^1):
				pg.draw.rect(screen, bg_white, (x*size, y*size, size, size))
		posArray.append(tempPos)
	
	soln = Solver.solver([], n)
	
	if soln:
		drawSoln(screen, soln, posArray)
	else:
		screen = pg.display.set_mode((800,400))
		font = pg.font.Font(None, 32)
		surf = font.render("Solution does not Exist!", True, (255,255,255), (0,0,0))
		rect = surf.get_rect()
		rect.center = (400, 200)
		screen.blit(surf,rect)
	
	pg.display.update()
	
	while not done:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				done = True



if __name__=="__main__":
	pg.init()
	main()
	pg.quit()