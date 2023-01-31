
#include "seismicsurvey.h"

#include <freehorizonattributrepfactory.h>
#include <freehorizonattribut.h>

FreeHorizonAttribut::FreeHorizonAttribut(WorkingSetManager * workingSet, SeismicSurvey *survey, const QString &path, const QString &name, QObject *parent)
	:IData(workingSet, parent), m_name(name){
	m_survey = survey;
	m_path = path;
	m_name = name;
	m_workingSet = workingSet;
	m_repFactory = new FreeHorizonAttributRepFactory(this);
}

FreeHorizonAttribut::~FreeHorizonAttribut()
{

}


QUuid FreeHorizonAttribut::dataID() const {
	return m_uuid;
}


IGraphicRepFactory *FreeHorizonAttribut::graphicRepFactory()
{
	return m_repFactory;
}


QString FreeHorizonAttribut::getPath()
{
	return m_path;
}

QString FreeHorizonAttribut::getName()
{
	return m_name;
}

SeismicSurvey *FreeHorizonAttribut::getSurvey()
{
	return m_survey;
}


WorkingSetManager* FreeHorizonAttribut::getWorkingSetManager()
{
	return m_workingSet;
}

