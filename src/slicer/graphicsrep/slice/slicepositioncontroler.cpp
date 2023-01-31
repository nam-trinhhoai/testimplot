#include "slicepositioncontroler.h"
#include "idata.h"
#include "slicerep.h"

SlicePositionControler::SlicePositionControler(SliceRep *rep, QObject *parent) :
	DataControler(parent) {
	m_rep = rep;
	m_dir = SliceDirection::Inline;
	m_position = rep->currentSliceWorldPosition();
	m_color = QColor(Qt::cyan);
	connect(m_rep, SIGNAL(sliceWordPositionChanged(int)), this,
						SLOT(setPositionFromRep(int)));
}

SlicePositionControler::~SlicePositionControler() {

}

void SlicePositionControler::requestPosChanged(int val)
{
	m_rep->setSliceWorldPosition(val);
}

QUuid SlicePositionControler::dataID()const {
	return m_rep->data()->dataID();
}

SliceDirection SlicePositionControler::direction() const {
	return m_dir;
}

int SlicePositionControler::position() const {
	return m_position;
}

void SlicePositionControler::setPositionFromRep(int val) {
	m_position = val;
	emit posChanged(val);
}

QColor SlicePositionControler::color()const{
	return m_color;
}

void SlicePositionControler::setColor(const QColor &c)
{
	m_color=c;
}

void SlicePositionControler::setDirection(SliceDirection dir)
{
	m_dir=dir;
}

