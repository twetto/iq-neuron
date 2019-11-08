#ifndef WEIGHT_INDEX_LIST_H
#define WEIGHT_INDEX_LIST_H
#include <stdio.h>

class iq_network;
class weight_index_list;

class weight_index_node
{
public:
    weight_index_node(int post);
    
    int _post;
    weight_index_node *_next;
};

class weight_index_list
{
public:
    weight_index_list();
    ~weight_index_list();
    void push_front(int post);

    weight_index_node *_first;
};

#endif

