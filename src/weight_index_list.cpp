/* Singly-linked list for axon indexing
 * Chen-Fu Yeh, 2019/11/09
 */

#include "weight_index_list.h"

using namespace std;

weight_index_node::weight_index_node(int data)
{
    _data = data;
    _next = NULL;
    return;
}

weight_index_list::weight_index_list()
{
    _first = NULL;
    return;
}

weight_index_list::~weight_index_list()
{
    while(_first != NULL) {
        weight_index_node *current = _first;
        _first = _first->_next;
        delete current;
    }
    return;
}

void weight_index_list::push_front(int data)
{
    weight_index_node *newNode = new weight_index_node(data);
    newNode->_next = _first;
    _first = newNode;
    return;
}

