# Joey Velez-Ginorio
# Script to generate stimuli / paths
# -----------------------------------------
from Tkinter import *
import turtle
import itertools
import numpy as np 
import math

class makeStimuli():
	"""
		Generates all the images for a stimuli set.

	"""
	
	def __init__(self):

		objectList = ['object1small.gif','object2small.gif','object3small.gif']
		self.wn = turtle.Screen()
		turtle.tracer(0,0)
		turtle.pensize(10)
		self.wn.setup(1800,800)

		self.coords = [
		{'S':(-200,-300), '1':(-800,300),'2':(200,-300),'3':(800,300),'4':(0,100),
		'5':(-200,-100),'6':(200,-100), '7':(-200,300), '8':(200,300)},
		]
		# {'1':(-350,-350), '2':(-350,350),'3':(350,350),'S':(350,-350)},
		# {'2':(-350,-350), '3':(-350,350),'S':(350,350),'1':(350,-350)},
		# {'3':(-350,-350), 'S':(-350,350),'1':(350,350),'2':(350,-350)}
		# ]

		# self.coords = [
		# {'S':(-416,-416), '1':(-416,416),'2':(416,416),'3':(416,-416),'4':(0,0)},
		# {'S':(-416,416), '1':(416,416),'2':(416,-416),'3':(-416,-416),'4':(0,0)},
		# {'S':(416,416), '1':(416,-416),'2':(-416,-416),'3':(-416,416),'4':(0,0)},
		# {'S':(416,-416), '1':(-416,-416),'2':(-416,416),'3':(416,416),'4':(0,0)},
		# ]

		# self.coords = [
		# {'S':(0,-416), '1':(-416,416),'2':(0,416),'3':(416,416),'4':(0,0)}
		# # {'S':(-416,416), '1':(416,416),'2':(416,-416),'3':(-416,-416),'4':(0,0)},
		# # {'S':(416,416), '1':(416,-416),'2':(-416,-416),'3':(-416,416),'4':(0,0)},
		# # {'S':(416,-416), '1':(-416,-416),'2':(-416,416),'3':(416,416),'4':(0,0)},
		# ]
		self.allObjectLists = list(itertools.permutations(objectList,3))
		self.allPaths = list()
		for i in range(1,4):
			self.allPaths += list(itertools.permutations('12',i))
		self.allPaths = [list(i) for i in self.allPaths]
		# t = self.allPaths
		self.allPaths = [
		# A, B, C, AB, AC, BA, BC, CA, CB
		['1'],
		['5','4','6','2'],
		['5','4','8','3'],
		['1','7','4','6','2'],
		['1','3'],
		['5','4','6','2','6','7','1'],
		['5','4','6','2','3'],
		['5','8','3','1'],
		['5','8','3','2'],
		]
		
		# temp = [t[0],t[1],t[3],t[4],t[5],t[7],t[8],t[15],t[16],t[18],t[22],t[24],t[38],t[39],t[62],t[63]]
		# self.allPaths = temp
		
		# for i in range(len(self.allPaths)):
		# 	if '3' in self.allPaths[i]:
		# 		ind = self.allPaths[i].index('3')
		# 		self.allPaths[i].insert(ind,'4')


		for k in range(len(self.coords)):

			for i in range(1):
				last=False
				for j in range(len(self.allPaths)):
					self.setup(self.wn, self.allObjectLists[i], self.coords[k],k)
					self.wn.turtles()[0].ht()
					self.drawStimulus(self.wn, self.allPaths[j], self.coords[k])
					self.agent.ht()
					self.agentArrow.ht()
					# self.agentTracer.ht()
					turtle.update()
					ts = turtle.getscreen()
					ts.getcanvas().postscript(file="batch1/stim"+str(k)+"_"+"set"+str(i)+"_"+"path"+str(j)+".eps")
					self.wn.clear()
					turtle.tracer(0,0)

		turtle.bye()

	def setup(self, wn, objectList, coord,k):

		wn.addshape('agentsmall.gif')


		self.grid = turtle.Turtle()
		self.grid.up()
		self.grid.goto(-900+200,-400)
		self.grid.down()
		self.grid.goto(-900+200,400) #1
		self.grid.goto(-900+200+200,400)
		self.grid.goto(-900+200+200,-400) #2
		self.grid.goto(-900+200+200+200,-400)
		self.grid.goto(-900+200+200+200,400) #3
		self.grid.goto(-900+200+200+200+200,400)
		self.grid.goto(-900+200+200+200+200,-400) #4
		self.grid.goto(-900+200+200+200+200+200,-400) 
		self.grid.goto(-900+200+200+200+200+200, 400) #5
		self.grid.goto(-900+200+200+200+200+200+200, 400) 
		self.grid.goto(-900+200+200+200+200+200+200, -400) #6
		self.grid.goto(-900+200+200+200+200+200+200+200, -400)
		self.grid.goto(-900+200+200+200+200+200+200+200, 400) #7
		self.grid.goto(-900+200+200+200+200+200+200+200+200, 400)
		self.grid.goto(-900+200+200+200+200+200+200+200+200, -400) #8

		self.grid.up()
		self.grid.goto(-900,-400+200)
		self.grid.down()
		self.grid.goto(900,-400+200) #1
		self.grid.goto(900,-400+200+200)
		self.grid.goto(-900,-400+200+200) #2
		self.grid.goto(-900,-400+200+200+200)
		self.grid.goto(900,-400+200+200+200) #3
		self.grid.goto(900,-400+200+200+200+200)
		self.grid.goto(-900,-400+200+200+200+200) #4

		self.grid.ht()

		# if k == 0:
		self.wall = turtle.Turtle()
		self.wall.pensize(2)
		self.wall.penup()
		self.wall.goto(-40,-400)
		self.wall.pendown()
		self.wall.begin_fill()
		self.wall.forward(80)
		self.wall.left(90)
		self.wall.forward(400)
		self.wall.left(90)
		self.wall.forward(80)
		self.wall.left(90)
		self.wall.forward(400)
		self.wall.color('gray')
		self.wall.end_fill()
		self.wall.ht()

		self.border = turtle.Turtle()
		self.border.pensize(6)
		self.border.penup()
		self.border.goto(-900,-400)
		self.border.pendown()
		self.border.goto(-900,400)
		self.border.goto(896,400)
		self.border.goto(896,-396)
		self.border.goto(-896,-396)
		# if k == 1:
		# 	self.wall = turtle.Turtle()
		# 	self.wall.pensize(2)
		# 	self.wall.penup()
		# 	self.wall.goto(-900,-20)
		# 	self.wall.pendown()
		# 	self.wall.begin_fill()
		# 	self.wall.forward(416)
		# 	self.wall.left(90)
		# 	self.wall.forward(40)
		# 	self.wall.left(90)
		# 	self.wall.forward(416)
		# 	self.wall.left(90)
		# 	self.wall.forward(40)
		# 	self.wall.color('gray')
		# 	self.wall.end_fill()
		# 	self.wall.ht()
		# if k == 2:
		# 	self.wall = turtle.Turtle()
		# 	self.wall.pensize(2)
		# 	self.wall.penup()
		# 	self.wall.goto(-20,900)
		# 	self.wall.pendown()
		# 	self.wall.begin_fill()
		# 	self.wall.forward(40)
		# 	self.wall.right(90)
		# 	self.wall.forward(416)
		# 	self.wall.right(90)
		# 	self.wall.forward(40)
		# 	self.wall.right(90)
		# 	self.wall.forward(416)
		# 	self.wall.color('gray')
		# 	self.wall.end_fill()
		# 	self.wall.ht()
		# if k == 3:
		# 	self.wall = turtle.Turtle()
		# 	self.wall.pensize(2)
		# 	self.wall.penup()
		# 	self.wall.goto(900,-20)
		# 	self.wall.pendown()
		# 	self.wall.begin_fill()
		# 	self.wall.left(180)
		# 	self.wall.forward(416)
		# 	self.wall.right(90)
		# 	self.wall.forward(40)
		# 	self.wall.right(90)
		# 	self.wall.forward(416)
		# 	self.wall.right(90)
		# 	self.wall.forward(40)
		# 	self.wall.color('gray')
		# 	self.wall.end_fill()
		# 	self.wall.ht()


		# Create agent in bottom left corner
		self.agent = turtle.Turtle('agentsmall.gif')
		self.agent.penup()
		# self.agent.ht()
		self.agent.goto(coord['S'][0], coord['S'][1])
		self.agent.dot(15)
		if self.agent.position()[1] < 0:
			self.agent.goto(coord['S'][0], coord['S'][1]-60)
			self.agent.write("START",align="center",font=("Arial",40,"bold"))
		else:
			self.agent.goto(coord['S'][0], coord['S'][1]-60)
			self.agent.write("START",align="center",font=("Arial",40,"bold"))


		# Assign .gif paths to objectsgf
		object1 = objectList[0]
		object2 = objectList[1]
		object3 = objectList[2]

		# Create object 1 at top left corner
		wn.addshape(object1)
		self.object1 = turtle.Turtle(object1)
		self.object1.penup()
		self.object1.goto(coord['1'][0], coord['1'][1])
		self.object1.resizemode('user')
		self.object1.shapesize(5,5,12)
		# self.object1.ht()
		# self.object1.dot(10)
		self.object1name = turtle.Turtle()
		self.object1name.ht()
		self.object1name.penup()
		self.object1name.goto(coord['1'][0]+50,coord['1'][1]+0)
		self.object1name.write("A",align="left",font=("Arial",50,"bold"))
		# self.object1.ht()

		# Create object 2 at top right corner
		wn.addshape(object2)
		self.object2 = turtle.Turtle(object2)
		self.object2.penup()
		self.object2.goto(coord['2'][0],coord['2'][1])
		# self.object2.ht()
		# self.object2.dot(200)
		# self.object2.ht()
		self.object2name = turtle.Turtle()
		self.object2name.ht()
		self.object2name.penup()
		self.object2name.goto(coord['2'][0]+50,coord['2'][1]+0)
		self.object2name.write("B",align="left",font=("Arial",50,"bold"))

		# Create object 3 at bottom right corner
		wn.addshape(object3)
		self.object3 = turtle.Turtle(object3)
		self.object3.penup()
		self.object3.goto(coord['3'][0], coord['3'][1])
		# self.object3.ht()
		# self.object3.dot(200)

		# self.object3.ht()
		self.object3name = turtle.Turtle()
		self.object3name.ht()
		self.object3name.penup()
		self.object3name.goto(coord['3'][0]+50,coord['3'][1]-0)
		self.object3name.write("C",align="left",font=("Arial",50,"bold"))

		# Create invisible agent that will be used to trace paths
		self.agentTracer = turtle.Turtle('agentsmall.gif')
		self.agentTracer.ht()
		self.agentTracer.penup()
		self.agentTracer.goto(coord['S'][0], coord['S'][1])
		self.agentTracer.pendown()

		self.agentArrow = turtle.Turtle('arrow')
		self.agentArrow.turtlesize(1.0,1.0,1.0)
		self.agentArrow.ht()
		self.agentArrow.penup()

	def drawStimulus(self, wn, path, coords):
		self.agentTracer.pensize(3)
		last=False
		t = False
		for i in range(len(path)):
			# self.agentTracer.goto(coords[spot][0], coords[spot][1])
			if i==len(path)-1:
				last=True
				nextSpot = None
			else:
				nextSpot = path[i+1]
			if i > 0 and ((path[i-1]=='2' and path[i]=='6' and path[0] == '5') or
			(path[i-1]=='3' and path[i]=='1' and path[0] == '5') ):
				t = True
				coords = self.changeCoords(coords,20)

			if i==0:
				first=True
			else:
				first=False


			self.drawPath(path[i], coords,last,t,nextSpot,first)
			t = False

		self.agentTracer.st()
			# self.agent.goto(self.coords[path[-1]][0], self.coords[path[-1]][0])
			# self.object1.ht()
			# self.object1.st(),
			# self.object2.ht()
			# self.object2.st()

	def changeCoords(self, coords, off):
		temp = [list(i) for i in coords.values()]
		temp = [(i[0]+off,i[1]+off) for i in temp]
		return dict(zip(coords.keys(),temp))


	def getAngle(self,current, target):
		y = target[1] - current[1]
		x = target[0] - current[0]
		if x == 0:

			return 90.0 if y > 0 else -90.0

		elif y == 0:
			return 0.0 if x > 0 else -180.0
		
		elif current[0] > target[0]:
			return 180.0 + np.arctan(y/x)*180/math.pi
		elif current[0] < target[0]:
			return np.arctan(y/float(x))*180/math.pi


	def drawPath(self, spot, coords,last,t,nextSpot, first):
		if t:
			self.agentTracer.up()
			self.agentTracer.goto(self.agentTracer.position()[0]+20,self.agentTracer.position()[1]+20)
			self.agentTracer.down()

		angle = self.agentTracer.towards(coords[spot])
		xoffset = coords[spot][0] - self.agentTracer.position()[0]
		yoffset = coords[spot][1] - self.agentTracer.position()[1]

		if xoffset == 0:
			xoffset = 0
		elif xoffset > 0:
			if yoffset != 0:
				xoffset = 2 * abs(xoffset/float(yoffset))
			else:
				xoffset = 2 
		else:
			if yoffset != 0:
				xoffset = -2 * abs(xoffset/float(yoffset))
			else:
				xoffset = -2
	
		if yoffset == 0:
			yoffset = 0

		elif yoffset > 0:
			yoffset = 2
		else:
			yoffset = -2

		# print 'here'
		i = 0
		counter = 0

		print xoffset, yoffset
		print coords[spot][0], coords[spot][1]
		print self.agentTracer.position()[0],self.agentTracer.position()[1]

		count = 0
		while(self.agentTracer.position() != coords[spot]):
			curr = self.agentTracer.position()
			curr = (curr[0],curr[1])

			if self.agentTracer.distance(coords[spot]) < 2:
					self.agentTracer.setheading(angle)
					self.agentTracer.goto(curr[0]+xoffset, curr[1]+yoffset)
					break

			if last:
				# print 'here'
				if abs(curr[0]-coords[spot][0] == 0):
					if abs(curr[1]-coords[spot][1]) < 200:
						break
				elif abs(curr[1]-coords[spot][1] == 0):
					if abs(curr[0]-coords[spot][0]) < 200:
						break
				elif abs(curr[0]-coords[spot][0])+abs(curr[1]-coords[spot][1]) < 400:
					print 'here'
					break

			# print curr, spot, coords[spot]
			self.agentTracer.setheading(angle)
			self.agentTracer.goto(curr[0]+xoffset, curr[1]+yoffset)
			self.agentTracer.penup()


			curr = self.agentTracer.position() 

			# print 'i: ',i
			if i % 50 == 0:
				self.agentArrow.setheading(angle)
				self.agentArrow.goto(curr[0],curr[1])
				count += 1

				if last:
					if count == 20000:
						# self.agentArrow.stamp()
						pass
					else:
						if first:
							pass
						else:
							self.agentArrow.stamp()

				else:
					if first:
						first=False

					else:
						self.agentArrow.stamp()

			self.agentTracer.goto(curr[0]+xoffset, curr[1]+yoffset)

			# self.agentTracer.forward(5)
			if i % 5 == 0:
				self.agentTracer.pendown()
			i+=1
			# if i == 400:
			# 	break



"""

Wingate et al, ICJAI - 2011

"""