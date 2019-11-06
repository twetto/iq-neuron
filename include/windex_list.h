#ifndef WINDEX_LIST_H
#define WINDEX_LIST_H
#include <stdio.h>

class windex_list;

class windex_node
{
public:
    windex_node();
    windex_node(int pre, int post);
    
    friend class windex_list;

private:
    int _pre, _post;
    windex_node *next;
}

class windex_list
{
public:
    windex_list();
    void push_front(int pre, int post);

private:
    windex_node *first;

};

#endif

