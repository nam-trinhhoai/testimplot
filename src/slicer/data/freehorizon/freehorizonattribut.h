
#ifndef __FREEHORIZONATTRIBUT__
#define __FREEHORIZONATTRIBUT__

#include <QObject>
#include <QVector2D>
#include <QList>
#include <QColor>
#include <vector>

#include "idata.h"

class FreeHorizonAttributRepFactory;
class FreeHorizonAttributRep;
class SeismicSurvey;

class FreeHorizonAttribut : public IData {
Q_OBJECT
public:
	FreeHorizonAttribut(WorkingSetManager * workingSet, SeismicSurvey *survey, const QString &path, const QString &name, QObject *parent);
	virtual  ~FreeHorizonAttribut();
	QString name() const override{return m_name;}
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;

	QString getPath();
	QString getName();
	SeismicSurvey *getSurvey();
	WorkingSetManager *getWorkingSetManager();


private:
	QString m_name;
	FreeHorizonAttributRepFactory *m_repFactory = nullptr;
	FreeHorizonAttributRep *m_layer = nullptr;
	QUuid m_uuid;
	SeismicSurvey *m_survey;
	QString m_path = "";
	WorkingSetManager *m_workingSet = nullptr;

signals:
	void freeHorizonAttributAdded();
};




#endif
