function new_labels = graph_cut(cost_per_node, pairwise_cost, vertical_cost, horizontal_cost, n_iters)
%
% pixels converted to nodes in column-major order
%
addpath('./gco-v3/matlab');
[H, W, num_classes] = size(cost_per_node);
n1 = zeros(2*(H-1)*(W-1) + H-1 + W-1, 1);
n2 = zeros(2*(H-1)*(W-1) + H-1 + W-1, 1);
values = zeros(2*(H-1)*(W-1) + H-1 + W-1, 1);
cntr = 1;
for j=1:W-1
    for i=1:H-1
        n1(cntr) = (j-1)*H+i; n1(cntr+1) = (j-1)*H+i;
        n2(cntr) = (j-1)*H+i+1; n2(cntr+1) = j*H+i;
        values(cntr) = horizontal_cost(i,j); values(cntr+1) = vertical_cost(i,j);
        cntr = cntr+2;
    end
end
for i=1:H-1
    n1(cntr) = (W-1)*H+i; n2(cntr) = (W-1)*H+i+1;
    values(cntr) = vertical_cost(i, W);
    cntr = cntr+1;
end
for j=1:W-1
    n1(cntr) = j*H; n2(cntr) = j*H+H;
    values(cntr) = horizontal_cost(H, j);
    cntr = cntr+1;
end
neighbour_weight_matrix = sparse(n1, n2, values, H*W, H*W);
% create general graph
graph = GCO_Create(H*W, num_classes);
% assign individual cost per node
GCO_SetDataCost(graph, reshape(cost_per_node, [H*W, num_classes])');
% assign pairwise label cost
GCO_SetSmoothCost(graph, pairwise_cost);
% assign neighbors and edge weights
GCO_SetNeighbors(graph, neighbour_weight_matrix);
% perform graph cut optimisation
GCO_Swap(graph, n_iters);
labels = GCO_GetLabeling(graph);
new_labels = reshape(labels, [H, W]);
GCO_Delete(graph);
rmpath('./gco-v3/matlab');
end