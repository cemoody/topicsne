In this script we will try to model the output t-SNE space in 
a topic model way. 

    0. Measure distances d_ij of points

    1. Construct the input p_ij

    2. Define qij = (1 + dij2)^-1  /  SUM(over k, l, k!=l) (1 + dkl2)^-1
              qij = (1 + dij2)^-1  /  (-N + SUM(over k, l) (1 + dkl2)^-1)
              qij = (1 + dij2)^-1  /  Z
          dij2 = ||x_i - x_j||^2
          Where x_i = gs(r_i) . M
          r_i = is a loading of a document onto topics
          M = translation from topics to vector space
          gs = gumbel-softmax of input rep

    3. Algorithm:
      3.a Precompute p_ij
      3.b Build pairwise matrix Sum dkl2
          For all points, sample x_i = gs(r_i) . M
          Build N^2 matrix of pairwise distances:  dkl2 = ||xk||^2 + ||xl||^2 - 2 xk . xl
          Z = Sum over all, then subtract N to compensate for diagonal entries
      3.c For input minibatch of ij, minimize p_ij (log(p_ij) - log(q_ij))
    3. SGD minimize p_ij log(p_ij / q_ij)
