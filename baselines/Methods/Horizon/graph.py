class Vertex:
    def __init__(self, key, key1, type):
        """
        Initialize a vertex object representing a node in the functional dependency
        pattern graph (typically an attribute value or attribute-value pair).

        :param key: Vertex ID, usually the identifier of an attribute or tuple.
        :param key1: Vertex attribute, representing the data attribute value or name.
        :param type: Vertex type, used to distinguish different vertex types in the FD
                     pattern graph (e.g., bound attribute vs. free attribute).
        """
        self.id = key  # unique identifier of the vertex
        self.attr = key1  # attribute name or attribute value
        self.type = type  # vertex type: 0 = bound attribute, 1 = free attribute
        self.connectedTo = {}  # connections to other vertices and their weights (adjacency)
        self.connectedQLT = {}  # connection quality to other vertices (edge weights for repair)

    def addNeighbor(self, nbr):
        """
        Add a connection to a neighbor vertex, representing a functional dependency
        relationship (from one attribute to another). If the neighbor already exists,
        accumulate the connection count.

        :param nbr: Neighbor vertex object connected to the current vertex in the FD
                    pattern graph.
        """
        if nbr in self.connectedTo:
            self.connectedTo[nbr] += 1  # neighbor already exists, increment weight
        else:
            self.connectedTo[nbr] = 1  # new neighbor, initialize weight to 1
        self.connectedQLT[nbr] = 0  # initialize connection quality to 0 (updated later)

    def __str__(self):
        """
        Return a string representation of the vertex showing its ID and connected neighbors.

        :return: string representation of the vertex
        """
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        """
        Get all neighbor vertices connected to the current vertex. These typically
        represent other attributes or attribute values in the functional dependency.

        :return: list of neighbor vertices connected to the current vertex
        """
        return list(self.connectedTo.keys())

    def getId(self):
        """
        Get the unique identifier of the vertex.

        :return: vertex ID
        """
        return self.id

    def getAttr(self):
        """
        Get the attribute name or value of the vertex, typically representing a
        specific attribute in the FD pattern.

        :return: vertex attribute
        """
        return self.attr

    def getType(self):
        """
        Get the vertex type, used to distinguish different types of nodes in the FD
        pattern graph (e.g., 0 = bound attribute, 1 = free attribute).

        :return: vertex type
        """
        return self.type

    def getWeight(self, nbr):
        """
        Get the connection weight to a neighbor vertex, representing the strength
        or influence of the functional dependency. In pattern repair, this weight
        reflects the frequency or priority with which a pattern is selected.

        :param nbr: neighbor vertex connected via a functional dependency.
        :return: connection weight to the neighbor vertex
        """
        return self.connectedTo[nbr]

class Graph:
    def __init__(self):
        """
        Initialize the graph object. This graph represents the functional dependency
        pattern graph, where vertices denote data attributes or attribute values and
        edges denote functional dependency relationships.

        self.vertList: dictionary storing the graph's vertices (key = vertex ID,
                       value = Vertex object).
        self.numVertices: integer recording the number of vertices in the graph.
        """
        self.vertList = {}  # dictionary of vertices (key: vertex ID, value: Vertex)
        self.numVertices = 0  # number of vertices in the graph

    def addVertex(self, key, key1, type):
        """
        Add a new vertex to the graph. A vertex typically represents an attribute or
        attribute value in the data.

        :param key: vertex ID (unique identifier)
        :param key1: vertex attribute (data attribute value represented by this vertex)
        :param type: vertex type, distinguishing bound and free attributes
        :return: the newly added vertex object
        """
        if key not in self.vertList:  # if the vertex ID is not in the graph, add it
            self.numVertices += 1  # increase vertex count
            newVertex = Vertex(key, key1, type)  # create the new vertex
            self.vertList[key] = newVertex  # add to the vertex list
            return newVertex  # return the newly added vertex

    def getVertex(self, n):
        """
        Retrieve the vertex with the specified ID.

        :param n: vertex ID
        :return: the vertex object if found, otherwise None
        """
        if n in self.vertList:  # if the vertex ID exists in the graph
            return self.vertList[n]  # return the corresponding vertex object
        else:
            return None  # vertex not found

    def __contains__(self, n):
        """
        Check whether the graph contains a vertex with the specified ID.

        :param n: vertex ID
        :return: True if the vertex exists, False otherwise
        """
        return n in self.vertList  # check whether the vertex is in the vertex list

    def addEdge(self, f, t, const=0):
        """
        Add an edge between two vertices, representing a functional dependency
        between two attributes.

        :param f: source vertex ID, representing the LHS (premise attribute) of the FD
        :param t: target vertex ID, representing the RHS (consequent attribute) of the FD
        :param const: optional parameter, default 0, representing the edge's default
                      weight (can denote dependency strength)
        """
        if f in self.vertList and t in self.vertList:  # both vertices must be in the graph
            self.vertList[f].addNeighbor(self.vertList[t])  # add the edge

    def getVertices(self):
        """
        Get all vertex IDs in the graph.

        :return: list of vertex IDs
        """
        return self.vertList.keys()  # return all vertex IDs

    def __iter__(self):
        """
        Implement the iterator protocol, allowing iteration over all vertex objects.

        :return: iterator over the graph's vertex objects
        """
        return iter(self.vertList.values())  # return iterator over vertex objects
def tr(G):
    """
    Compute the transpose of the graph (reverse the direction of every edge).

    :param G: graph adjacency-list representation. G is a dict whose keys are vertices
              and whose values are sets of neighbors that the vertex points to.
    :return: transpose graph adjacency-list representation. GT is a dict whose keys are
             vertices and whose values are sets of neighbors that point to the vertex.
    """
    GT = dict()  # new dictionary to store the transpose graph
    for u in G.keys():
        GT[u] = GT.get(u, set())  # initialize every vertex with an empty neighbor set
    for u in G.keys():
        for v in G[u]:
            GT[v].add(u)  # reverse every edge (u -> v) to (v -> u)
    return GT  # return the transpose graph

def topoSort(G):
    """
    Topologically sort a directed acyclic graph (DAG).

    :param G: graph adjacency-list representation. G is a dict whose keys are vertices
              and whose values are sets of neighbors that the vertex points to.
    :return: list of vertices in topological order.
    """
    res = []  # stores the topological order
    S = set()  # stores visited vertices

    def dfs(G, u):
        """
        Depth-first search helper.

        :param G: graph adjacency list
        :param u: current vertex
        """
        if u in S:  # if vertex u has already been visited, return immediately
            return
        S.add(u)  # mark vertex u as visited
        for v in G[u]:  # traverse neighbors of u
            if v in S:
                continue
            dfs(G, v)  # recursive DFS
        res.append(u)  # append u to result after DFS finishes

    for u in G.keys():
        dfs(G, u)  # run DFS from every vertex
    res.reverse()  # reverse the result to obtain the topological order
    return res  # return the topological order


def walk(G, s, S=None):
    """
    Traverse the graph starting from vertex s and return the path. Optionally ignore
    some vertices.

    :param G: graph adjacency-list representation. G is a dict whose keys are vertices
              and whose values are sets of neighbors that the vertex points to.
    :param s: starting vertex
    :param S: optional set of vertices to ignore during traversal. Defaults to None.
    :return: dictionary P representing the path starting at s; keys are vertices and
             values are their predecessors.
    """
    if S is None:
        S = set()  # if S is not specified, use an empty set
    Q = []  # list used as a stack (or queue)
    P = dict()  # path dictionary: key = vertex, value = predecessor
    Q.append(s)  # enqueue the starting vertex
    P[s] = None  # starting vertex has no predecessor

    while Q:
        u = Q.pop()  # pop the last element
        for v in G[u]:  # traverse neighbors of u
            if v in P.keys() or v in S:  # skip if already visited or ignored
                continue
            Q.append(v)  # enqueue neighbor v
            P[v] = u  # set u as predecessor of v
    return P  # return the path dictionary
