import numpy as np

class GNG():
  def __init__(self, epsb = 0.1, epsn = 0.01, ageMax = 10, interval = 100, discount = 0.99, alpha = 0.5):
    self.positions = np.array( [[0.,0.],[1.,1.]] )
    self.errors = np.zeros(2)
    self.edges = np.array( [[0,1]] )
    self.edgeAge = np.array( [0] )
    self.epsb = epsb
    self.epsn = epsn
    self.ageMax = ageMax
    self.interval = interval
    self.samples = 0
    self.discount = discount
    self.alpha = alpha

  def printState (self):
    print("positions " + str(self.positions))
    print("edges " + str(self.edges))
    print("ages " + str(self.edgeAge))
    print("errors " + str(self.errors))
    
  def addSample(self, position):

    sqrDistances = np.zeros (self.positions.shape[0])
    for i in range(self.positions.shape[0]):
      v = position - self.positions[i,:]
      sqrDistances[i] = v.dot(v)

    # find first and second distances
    prInd = np.argmin(sqrDistances)
    prDist = sqrDistances[prInd]
    sqrDistances[prInd] = np.inf
    scInd = np.argmin(sqrDistances)
    scDist = sqrDistances[scInd]
    sqrDistances[scInd] = np.inf

    # increase the age of all edges attached to pr
    prEdges = np.where (self.edges == prInd)[0]
    self.edgeAge[prEdges] += 1 # increment the error
    self.errors[prInd] += prDist

    # shift prim position toward the sample by epsb
    prDir = position - self.positions[prInd]
    self.positions[prInd] += self.epsb * prDir

    # shift the neighbors toward it by epsn
    hasScEdge = False
    for e in prEdges:
      edge = self.edges[e]
      nInd = -1
      if (edge[0] == prInd):
        nInd = edge[1]
      else:
        nInd = edge[0]
      nDir = position - self.positions[nInd]
      self.positions[nInd] += self.epsn * nDir

      # set the edge age between pr and sc to 0
      if (nInd == scInd):
        self.edgeAge[e] = 0
        hasScEdge = True

    # no edge between pr and sc, add it
    if not hasScEdge:
      sz = len(self.edges)
      self.edges.resize(sz+1, 2, refcheck=False)
      self.edgeAge.resize(sz+1, refcheck=False)
      self.edges[sz] = np.array([prInd,scInd])
      self.edgeAge[sz] = 0

    young = np.where(self.edgeAge < self.ageMax)
    self.edges = self.edges[young]
    self.edgeAge = self.edgeAge[young]

    self.samples += 1    
    # at interval split the highest error edge
    if self.samples >= self.interval:
      self.samples = 0
      # find edge with highest combined error
      maxError = self.errors[self.edges[0,0]] + self.errors[self.edges[0,1]]
      maxEdge = 0 
      for e in range(len(self.edges)):
        error = self.errors[self.edges[e,0]] + self.errors[self.edges[e,1]]
        if error > maxError:
          maxError = error
          maxEdge = e

      sz = len(self.positions)
      self.positions.resize(sz+1, 2, refcheck=False)
      self.errors.resize(sz+1, refcheck=False)
      
      # new position half-way between with error equal to highest of edge
      self.positions[sz] = 0.5 * (self.positions[self.edges[maxEdge,0]] + self.positions[self.edges[maxEdge,1]])
      self.errors[sz] = max (self.errors[self.edges[maxEdge,0]], self.errors[self.edges[maxEdge,1]])
      newPos = sz

      # discount edge errors both by alpha
      self.errors[self.edges[maxEdge,0]] *= self.alpha
      self.errors[self.edges[maxEdge,1]] *= self.alpha

      # delete old edge, make 2 new
      sz = len(self.edges)
      self.edges.resize(sz+1, 2, refcheck=False)
      self.edgeAge.resize(sz+1, refcheck=False)
      self.edges[sz,0] = newPos
      self.edges[sz,1] = self.edges[maxEdge,1]
      self.edges[maxEdge,1] = newPos
      self.edgeAge[sz] = 0
      self.edgeAge[maxEdge] = 0

    # reduce all errors by factor d < 1
    self.errors *= self.discount
