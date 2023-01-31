#ifndef Marker_H
#define Marker_H

#include <QObject>
#include <QVector2D>
#include <QList>
#include <QColor>

#include "idata.h"
#include "RgtLayerProcessUtil.h"

class WellPick;
class WellBore;
class MarkerGraphicRepFactory;
class Seismic3DAbstractDataset;

class Marker: public IData {
Q_OBJECT
public:
	Marker(WorkingSetManager * workingSet,const QString &name, QObject *parent =
			0);
	virtual ~Marker();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	QList<WellPick*> getWellPickFromWell(WellBore* bore);

	const QList<WellPick*>& wellPicks() const;

	QColor color() const;
	void setColor(const QColor& color);

	QList<RgtSeed> getProjectedPicksOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit);

public slots:
	void addWellPick(WellPick* pick);
	void removeWellPick(WellPick* pick);

signals:
	void wellPickAdded(WellPick* pick);
	void wellPickRemoved(WellPick* pick);

private:
	QString m_name;
	QUuid m_uuid;

	MarkerGraphicRepFactory* m_repFactory;
	QList<WellPick*> m_wellPicks;
	QColor m_color;
};

#endif
