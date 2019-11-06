#include "windex_list.h"

using namespace std;

windex_node::windex_node()
{
    _pre = 0;
    _post = 0;
    next = NULL;
    return;
}

windex_node::windex_node(int pre, int post)
{
    _pre = pre;
    _post = post;
    next = NULL;
    return;
}

windex_list::windex_list()
{
    first = NULL;
    return;
}

void windex_list::push_front(int pre, int post)
{
    windex_node *newNode = new windex_node(pre, post);
    newNode->next = first;
    first = newNode;
    return;
}

void windex_list::clear()
{
    while(first != NULL) {
        windex_node *current = first;
        first = first->next;
        delete current;
    }
    return;
}

