/* Singly-linked list for axon indexing
 * Chen-Fu Yeh, 2019/11/09
 */

#ifndef WEIGHT_INDEX_LIST_H
#define WEIGHT_INDEX_LIST_H
#include <stdio.h>

class weight_index_list;

class weight_index_node
{
public:
    weight_index_node(int data);
    
    int _data;
    weight_index_node *_next;
};

class weight_index_list
{
public:
    weight_index_list();
    ~weight_index_list();
    void push_front(int data);

    weight_index_node *_first;
};

#endif

