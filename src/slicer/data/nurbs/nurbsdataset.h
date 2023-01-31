#ifndef NURBSDATASET_H
#define NURBSDATASET_H

#include <QObject>
#include <QVector2D>
#include <QList>
#include <QColor>

#include "idata.h"
#include "manager.h"

class Manager;

class NurbsGraphicRepFactory;

class NurbsDataset: public IData
{
Q_OBJECT
public:
NurbsDataset(WorkingSetManager * workingSet,Manager* nurbs, const QString &name, QObject *parent =0);
	virtual ~NurbsDataset();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	Manager* getNurbs3d();
//	QList<WellPick*> getWellPickFromWell(WellBore* bore);

//	const QList<WellPick*>& wellPicks() const;

//	QColor color() const;
//	void setColor(const QColor& color);

//	QList<RgtSeed> getProjectedPicksOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit);

public slots:
//	void addWellPick(WellPick* pick);
//	void removeWellPick(WellPick* pick);

signals:
//	void wellPickAdded(WellPick* pick);
//	void wellPickRemoved(WellPick* pick);

private:
	QString m_name;
	QUuid m_uuid;


	Manager* m_nurbs= nullptr;

	NurbsGraphicRepFactory* m_repFactory;
//	QList<WellPick*> m_wellPicks;
//	QColor m_color;
};

#endif
