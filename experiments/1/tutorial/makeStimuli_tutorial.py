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
		self.wn.setup(1050,1050)

		# self.coords = [
		# {'S':(0,-350), '1':(-350,-350),'2':(0,350),'3':(350,-350)},
		# ]
		# {'1':(-350,-350), '2':(-350,350),'3':(350,350),'S':(350,-350)},
		# {'2':(-350,-350), '3':(-350,350),'S':(350,350),'1':(350,-350)},
		# {'3':(-350,-350), 'S':(-350,350),'1':(350,350),'2':(350,-350)}
		# ]


		self.coords = [
		{'S':(0,-416), '1':(-416,-416),'2':(416,-416),'3':(0,416), '4':(-416,0),'5':(416,0)}
		# {'S':(-416,416), '1':(416,416),'2':(416,-416),'3':(-416,-416)},
		# {'S':(416,416), '1':(416,-416),'2':(-416,-416),'3':(-416,416)},
		# {'S':(416,-416), '1':(-416,-416),'2':(-416,416),'3':(416,416)},
		]
		# self.coords = [
		# {'S':(0,-416), '1':(-416,416),'2':(0,416),'3':(416,416),'4':(0,0)}
		# # {'S':(-416,416), '1':(416,416),'2':(416,-416),'3':(-416,-416),'4':(0,0)},
		# # {'S':(416,416), '1':(416,-416),'2':(-416,-416),'3':(-416,416),'4':(0,0)},
		# # {'S':(416,-416), '1':(-416,-416),'2':(-416,416),'3':(416,416),'4':(0,0)},
		# ]
		self.allObjectLists = list(itertools.permutations(objectList,3))
		self.allPaths = list()
		for i in range(1,5):
			self.allPaths += list(itertools.permutations('123',i))
		self.allPaths = [list(i) for i in self.allPaths]
		# t = self.allPaths
		self.allPaths = [
		# A, B, AB, BA, AC, ABC, ACB, BAC
		# (A,C), (AB,CB), (BA,CA), (AC,CA), (A,A), (AB,AB), (BA,BA), (AC,AC), (BA,AC), (BA,BC), (AC,BC)
		['3'],
		['2'],
		['1','4','3'],
		['1'],
		['2','5','3']
		]
		
		# temp = [t[0],t[1],t[3],t[4],t[5],t[7],t[8],t[15],t[16],t[18],t[22],t[24],t[38],t[39],t[62],t[63]]
		# self.allPaths = temp
		
		# for i in range(len(self.allPaths)):
		# 	if '3' in self.allPaths[i]:
		# 		ind = self.allPaths[i].index('3')
		# 		self.allPaths[i].insert(ind,'4')


		for k in range(len(self.coords)):

			for i in [2]:
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
					ts.getcanvas().postscript(file="stim"+str(k)+"_"+"set"+str(i)+"_"+"path"+str(j)+".eps")
					self.wn.clear()
					turtle.tracer(0,0)

		turtle.bye()

	def setup(self, wn, objectList, coord,k):

		wn.addshape('agentsmall.gif')

		self.border = turtle.Turtle()
		self.border.pensize(5)
		self.border.penup()
		self.border.goto(-520,-520)
		self.border.pendown()
		self.border.goto(-520,520)
		self.border.goto(520,520)
		self.border.goto(520,-520)
		self.border.goto(-520,-520)

		self.grid = turtle.Turtle()
		self.grid.up()
		self.grid.goto(-520+208,-520)
		self.grid.down()
		self.grid.goto(-520+208,520) #1
		self.grid.goto(-520+208+208,520)
		self.grid.goto(-520+208+208,-520) #2
		self.grid.goto(-520+208+208+208,-520)
		self.grid.goto(-520+208+208+208,520) #3
		self.grid.goto(-520+208+208+208+208,520)
		self.grid.goto(-520+208+208+208+208,-520) #4

		self.grid.up()
		self.grid.goto(-520,-520+208)
		self.grid.down()
		self.grid.goto(520,-520+208) #1
		self.grid.goto(520,-520+208+208)
		self.grid.goto(-520,-520+208+208) #2
		self.grid.goto(-520,-520+208+208+208)
		self.grid.goto(520,-520+208+208+208) #3
		self.grid.goto(520,-520+208+208+208+208)
		self.grid.goto(-520,-520+208+208+208+208) #4

		self.grid.ht()



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
		# self.object2.dot(150)
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
		# self.object3.dot(150)

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
			if i > 0 and ((path[i-1]=='3' and path[i]=='1' and path[0] != '2') ):
				t = True
				coords = self.changeCoords(coords,0)

			self.drawPath(path[i], coords,last,t)
			t = False

		self.agentTracer.st()
			# self.agent.goto(self.coords[path[-1]][0], self.coords[path[-1]][0])
			# self.object1.ht()
			# self.object1.st(),
			# self.object2.ht()
			# self.object2.st()

	def changeCoords(self, coords, off):
		temp = [list(i) for i in coords.values()]
		temp = [(i[0],i[1]+off) for i in temp]
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


	def drawPath(self, spot, coords,last,t):
		if t:
			self.agentTracer.up()
			self.agentTracer.goto(self.agentTracer.position()[0],self.agentTracer.position()[1])
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
		i = 1
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
				# # print 'here'
				# if abs(curr[0]-coords[spot][0] == 0):
				# 	if abs(curr[1]-coords[spot][1]) < 624:
				# 		break
				# elif abs(curr[1]-coords[spot][1] == 0):
				# 	if abs(curr[0]-coords[spot][0]) < 624:
				# 		break
				# elif abs(curr[0]-coords[spot][0])+abs(curr[1]-coords[spot][1]) < 1250:
				# 	print 'here'
				# 	break
				pass

			# print curr, spot, coords[spot]
			self.agentTracer.setheading(angle)
			self.agentTracer.goto(curr[0]+xoffset, curr[1]+yoffset)
			self.agentTracer.penup()


			curr = self.agentTracer.position() 

			if i % 52 == 0:
				self.agentArrow.setheading(angle)
				self.agentArrow.goto(curr[0],curr[1])
				count += 1

				if last:
					if count == 4:
						pass
					else:
						self.agentArrow.stamp()

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