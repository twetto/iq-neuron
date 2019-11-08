#include "weight_index_list.h"

using namespace std;

weight_index_node::weight_index_node(int post)
{
    _post = post;
    _next = NULL;
    return;
}

weight_index_list::weight_index_list()
{
    _first = NULL;
    return;
}

void weight_index_list::push_front(int post)
{
    weight_index_node *newNode = new weight_index_node(post);
    newNode->_next = _first;
    _first = newNode;
    return;
}

void weight_index_list::clear()
{
    while(_first != NULL) {
        weight_index_node *current = _first;
        _first = _first->_next;
        delete current;
    }
    return;
}

