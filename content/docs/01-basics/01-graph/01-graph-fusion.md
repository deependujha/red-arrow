---
title: Graph Fusion
type: docs
prev: docs/01-basics/01-graph
next: docs/01-basics/02-concepts
sidebar:
  open: false
weight: 12
---

- Let `u` and `v` be two distinct vertices in graph G.
- The fusion of the vertices u and v are treated as a single new vertex `w` and every edge which was incident on u and v, are now incident on `w`.
- After fusion, `number of vertices in the graph always decreases by one`.
    - If original graph has `n` vertices, after fusion it will have `n-1` vertices.
- But, the number of edges may increase or decrease or remain the same.

---

In all the below diagrams, we fuse vertices `B` and `C`.

### => Simple graph:

![fusion 1](/01-basics/graph/fusion-1.png)

### => Vertices reduced but edges remain same:

![fusion 2](/01-basics/graph/fusion-2.png)

### => Fusion creating self-loop:

![fusion 3](/01-basics/graph/fusion-3.png)

### => Fusion in graph with self-loop:

![fusion 4](/01-basics/graph/fusion-4.png)

---

Credits: [GeeksforGeeks](https://www.geeksforgeeks.org/dsa/fusion-operation-in-graph/)
