#############################################################################################################################

##       #### ########  ########     ###    ########  #### ########  ######
##        ##  ##     ## ##     ##   ## ##   ##     ##  ##  ##       ##    ##
##        ##  ##     ## ##     ##  ##   ##  ##     ##  ##  ##       ##
##        ##  ########  ########  ##     ## ########   ##  ######    ######
##        ##  ##     ## ##   ##   ######### ##   ##    ##  ##             ##
##        ##  ##     ## ##    ##  ##     ## ##    ##   ##  ##       ##    ##
######## #### ########  ##     ## ##     ## ##     ## #### ########  ######

#############################################################################################################################

import torch
import torch_geometric

#############################################################################################################################

 ######  ##          ###     ######   ######  ########  ######
##    ## ##         ## ##   ##    ## ##    ## ##       ##    ##
##       ##        ##   ##  ##       ##       ##       ##
##       ##       ##     ##  ######   ######  ######    ######
##       ##       #########       ##       ## ##             ##
##    ## ##       ##     ## ##    ## ##    ## ##       ##    ##
 ######  ######## ##     ##  ######   ######  ########  ######
        
#############################################################################################################################
### Equivariant Graph Convolutional Layer

class EquivariantGraphConvolutionalLayer(torch.nn.Module):
#
### Initialization
#
    def __init__(self, input_nf, output_nf, hidden_nf, edge_nf=0, node_nf=0, radial_nf=1, activation=torch.nn.Tanhshrink(), aggregation="sum"):
        super(EquivariantGraphConvolutionalLayer, self).__init__()
                        
        self.aggregation = aggregation

        self.EdgeMultiLayerPerceptron = torch.nn.Sequential(
            torch.nn.Linear(input_nf + input_nf + radial_nf + edge_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation
        )

        self.NodeMultiLayerPerceptron = torch.nn.Sequential(
            torch.nn.Linear(input_nf + hidden_nf + node_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, output_nf)
        )
        
        self.CoordMultiLayerPerceptron = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, 1),
        )

        self.EdgeAttention = torch.nn.Sequential(
            torch.nn.Linear(input_nf + input_nf + edge_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, 1),
            torch.nn.Sigmoid()
        )
#
### Perform operation using the average or sum
#
    def SegmentOperation(self, data, index, segments, option="mean"):
        
        index  = index.unsqueeze(-1).expand(-1, data.size(1))
        
        result_shape = (segments, data.size(1))
        
        result = data.new_full(result_shape, 0)        
        result.scatter_add_(0, index, data)
        
        if option == "mean":
            count = data.new_full(result_shape, 0)
            count.scatter_add_(0, index, torch.ones_like(data))

        if option == "mean": result = result/count.clamp(min=1.0)
        
        return result
#
### Operations on the edges
#
    def EdgeOperation(self, source, target, radial, edge_attr):
        
        agg = torch.cat([source, target, radial], dim=1) if edge_attr is None else torch.cat([source, target, radial, edge_attr], dim=1)
        cat = torch.cat([source, target], dim=1)         if edge_attr is None else torch.cat([source, target, edge_attr], dim=1)
            
        out = self.EdgeMultiLayerPerceptron(agg)
        
        out = out*self.EdgeAttention(cat)
            
        return out
#
### Operations on the nodes
#
    def NodeOperation(self, nodes, edges, edge_attr, node_attr):
        
        row, col = edges
        agg      = self.SegmentOperation(edge_attr, row, segments=nodes.size(0), option="sum")
        
        agg = torch.cat([nodes, agg], dim=1) if node_attr is None else torch.cat([nodes, agg, node_attr], dim=1)
            
        out = self.NodeMultiLayerPerceptron(agg)

        out = nodes + out
            
        return out
#
### Aggregation of features
#
    def CoordinateOperation(self, coord, edges, rij, edge_feat):
        
        row, col = edges
        
        trans    = rij*self.CoordMultiLayerPerceptron(edge_feat)
        
        agg      = self.SegmentOperation(trans, row, segments=coord.size(0), option=self.aggregation)
                    
        return coord + agg
#
### Coordinates transformation
#
    def RadialOperation(self, edges, coord, batch):

        
        row, col = edges
        rij      = coord[row] - coord[col]
        radial   = torch.sum(rij*rij, 1).unsqueeze(1)

        return radial, rij
#
### Forward pass
#
    def forward(self, nodes, coord, edges, edge_attr=None, node_attr=None, batch=None):

        row, col    = edges

        radial, rij = self.RadialOperation(edges, coord, batch)

        edge_feat   = self.EdgeOperation(nodes[row], nodes[col], radial, edge_attr)
        
        coord       = self.CoordinateOperation(coord, edges, rij, edge_feat)
        
        nodes       = self.NodeOperation(nodes, edges, edge_feat, node_attr)

        return nodes, coord
    
#############################################################################################################################
### Equivariant Graph Neural Network

class EquivariantGraphNeuralNetwork(torch.nn.Module):
#
### Initialization
#
    def __init__(self, nodes, edge_nf=0, node_nf=0, hidden_nf=4, out_nf=1, layers=1, activation=torch.nn.Tanhshrink(), aggregation="sum"):
        super(EquivariantGraphNeuralNetwork, self).__init__()
                
        self.hidden_nf = hidden_nf
        self.layers    = layers

        self.Embedding = torch.nn.Sequential(
            torch.nn.Linear(nodes, hidden_nf),
            activation
        )

        for i in range(layers):
            self.add_module(f"EGCL{i}",
                EquivariantGraphConvolutionalLayer(hidden_nf, hidden_nf, hidden_nf, edge_nf=edge_nf, node_nf=edge_nf, radial_nf=1, activation=activation, aggregation=aggregation)
            )

        self.NodeDecoding = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, hidden_nf)
        )

        self.GraphDecoding = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, out_nf)
        )
#
### Forward pass
#
    def forward(self, nodes, coords, edges, edge_attr=None, node_attr=None, batch=None, size=1):
                        
        nodes = self.Embedding(nodes)
        
        for i in range(self.layers):
            nodes, coords = self._modules[f"EGCL{i}"](nodes, coords, edges, edge_attr=edge_attr, node_attr=node_attr, batch=batch)

        nodes = self.NodeDecoding(nodes)
        
        nodes = nodes.view(-1, nodes.size(0), self.hidden_nf)
        
        nodes = torch_geometric.nn.global_add_pool(nodes, batch, size=size)

        nodes = nodes.squeeze(0)

        out   = self.GraphDecoding(nodes)
        
        return out

#############################################################################################################################
