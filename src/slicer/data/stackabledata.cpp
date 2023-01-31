#include "stackabledata.h"

StackableData::StackableData() {

}

StackableData::~StackableData() {

}

AbstractStack::AbstractStack(QObject* parent) : QObject(parent) {

}

AbstractStack::~AbstractStack() {

}

AbstractLabelStack::AbstractLabelStack(QObject* parent) : AbstractStack(parent) {

}

AbstractLabelStack::~AbstractLabelStack() {

}

StackClassType AbstractLabelStack::stackClassType() const {
	return StackClassType::LABEL;
}

AbstractRangeStack::AbstractRangeStack(QObject* parent) : AbstractStack(parent) {

}

AbstractRangeStack::~AbstractRangeStack() {

}

StackClassType AbstractRangeStack::stackClassType() const {
	return StackClassType::RANGE;
}
