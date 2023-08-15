class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.weighted_degree = 0.0

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def add_weighted_degree(self,num):
        self.weighted_degree += float(num)

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

        self.vert_dict[frm].add_weighted_degree(cost)
        self.vert_dict[to].add_weighted_degree(cost)

    def get_vertices(self):
        return self.vert_dict.keys()

if __name__ == '__main__':

    g = Graph()

    '''
    # noun phrase nodes
    g.add_vertex('Michale Jordan_noun')
    # g.add_vertex('Kurt Miller_noun')
    g.add_vertex('MCMC_noun')

    # entity nodes
    g.add_vertex('Michale Jordan (basketballer)_entity')
    g.add_vertex('Michale Jordan (professor)_entity')
    g.add_vertex('Michale Jordan (new)_entity')
    # g.add_vertex('Kurt Miller(sports)_entity')
    # g.add_vertex('Kurt Miller(new)_entity')
    g.add_vertex('MCMC(algo)_entity')
    g.add_vertex('MCMC(city)_entity')
    g.add_vertex('MCMC(new)_entity')

    # noun phrase - entity edges
    g.add_edge('Michale Jordan_noun', 'Michale Jordan (basketballer)_entity', 0.5)
    g.add_edge('Michale Jordan_noun', 'Michale Jordan (professor)_entity', 0.3)
    g.add_edge('Michale Jordan_noun', 'Michale Jordan (new)_entity', 0)
    # g.add_edge('Kurt Miller_noun', 'Kurt Miller(sports)_entity', 0.2)
    # g.add_edge('Kurt Miller_noun', 'Kurt Miller(new)_entity', 0)
    g.add_edge('MCMC_noun', 'MCMC(algo)_entity', 0.5)
    g.add_edge('MCMC_noun', 'MCMC(city)_entity', 0.4)
    g.add_edge('MCMC_noun', 'MCMC(new)_entity', 0)
    '''
    g.add_vertex('Michale Jordan_noun')
    g.add_vertex('MCMC_noun')
    g.add_vertex('c')
    g.add_vertex('d')
    g.add_vertex('e')
    g.add_vertex('f')

    g.add_edge('Michale Jordan_noun', 'b', 0.7)
    g.add_edge('Michale Jordan_noun', 'c', 9)
    g.add_edge('Michale Jordan_noun', 'f', 14)
    g.add_edge('b', 'c', 10)
    g.add_edge('b', 'd', 15)
    g.add_edge('c', 'd', 11)
    g.add_edge('c', 'f', 2)
    g.add_edge('d', 'e', 6)
    g.add_edge('e', 'f', 9)


    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print('( %s , %s, %.2f)'  % ( vid, wid, v.get_weight(w)))

    for v in g:
        print('g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()]))