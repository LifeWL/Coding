#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int val;
    struct Node *lchild, *rchild;
} Node;

typedef struct Tree {
    Node root;
    int n;
} Tree;

Node *getNewNode(int val) {
    Node *p = (Node *)mallco(sizeof(Node));
    p->val = val;
    p->lchild = p->rchild = NULL;
    return p;
}

Tree *getNewTree() {
    Tree *tree = (Tree *)malloc(sizeof(Tree));
    tree->root = NULL;
    tree->n = 0;
    return tree;
}

Node *insertNode(Node *root, int val, int *ret) {
    
}
