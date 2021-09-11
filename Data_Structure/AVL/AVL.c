#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key, h;
    struct Node *lchild, *rchild;
} Node;

Node _NIL, *NIL = &_NIL;
__arrtibute__((constructor))
void init_NIL(){
    NIL->key = 0;
    NIL->h =0;
    NIL->lchild = NIL->rchild = NULL;
    return ;
}

Node *getNewNode(int key) {
    Node *p = (Node *)malloc(sizeof(Node));
    p->key = key;
    p->lchild = p->rchild = NULL;
    p->h = 1;
    return p;
}

void UpdateHeight(Node *root) {
    int h1 = root->lchild->h;
    int h2 = root->rchild->h;
    root->h = (h1 > h2 ? h1 : h2) + 1;
    return ;
}

Node *left_rotate(Node *root) {
    Node *temp = root->rchild;
    root->rchild = temp->lchild;
    temp->lchild = root;
    UpdateHeight(root);
    UpdateHeight(temp);
    return temp;
}

Node *maintain(Node *root) {
    if (abs(root->lchild->h - root->rchild->h) <= 1) return root;
    if (root->lchild->h > root->rchild->h) {
        if (root->lchild->rchild->h > root->lchild->lchild->h) {
            root->lchild = left_rotate(root->lchild);
        }
        root = right_rotate(root);
    } else {
        if (root->rchild->lchild->h > root->rchild->rchild->h) {
            root->rchild = right_rotate(root->rchild);
        }
        root = left_rotate(root);
    }
    return root;
}

Node *insert(Node *root, int key) {
    if (root == NIL) return getNewNode(key);
}
