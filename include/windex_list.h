#ifndef WINDEX_LIST_H
#define WINDEX_LIST_H
#include <stdio.h>

class iq_network;
class windex_list;

class windex_node
{
public:
    windex_node();
    windex_node(int pre, int post);
    
    friend class windex_list;
    friend class iq_network;

private:
    int _pre, _post;
    windex_node *next;
};

class windex_list
{
public:
    windex_list();
    void push_front(int pre, int post);
    void clear();

    friend class iq_network;

private:
    windex_node *first;

};

#endif

