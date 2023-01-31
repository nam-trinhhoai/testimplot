#include "abstractgraphicrep.h"
#include "abstractinnerview.h"

AbstractGraphicRep::AbstractGraphicRep(AbstractInnerView *parent):QObject(parent)
{
	m_parent=parent;
	m_name="";
}

QString AbstractGraphicRep::name() const
{
	return m_name;
}
void AbstractGraphicRep::setName(const QString & name)
{
	m_name=name;
	emit nameChanged();
}

AbstractGraphicRep::~AbstractGraphicRep()
{

}
bool AbstractGraphicRep::canBeDisplayed()const{
	return true;
}
