#ifndef WellPick_H
#define WellPick_H

#include <QObject>
#include <QVector2D>

#include "idata.h"
#include "ifilebaseddata.h"
#include "wellbore.h"
#include "RgtLayerProcessUtil.h"

class Marker;
class WellPickGraphicRepFactory;
class Seismic3DAbstractDataset;

class WellPick: public IData, public IFileBasedData {
Q_OBJECT
public:
	WellPick(WellBore* parentWell, WorkingSetManager * workingSet,const QString &name, const QString& kind, double value,
			const QString& idPath, QObject *parent =
			0);
	virtual ~WellPick();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override;

	QString markerName() const;
	double value() const;
	QString kind() const;
	WellUnit kindUnit() const;

	WellBore* wellBore() const;
	Marker* currentMarker() const;
	void setCurrentMarker(Marker* marker);

	std::pair<RgtSeed, bool> getProjectionOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit);
	static std::pair<RgtSeed, bool> getProjectionOnDataset(Seismic3DAbstractDataset* dataset, SampleUnit sampleUnit, WellBore* wellBore, double unitVal, WellUnit wellUnit);

	// return nullptr if could not read file
	static WellPick* getWellPickFromDescFile(WellBore* parentWell, QColor color, QString descFile, WorkingSetManager * workingSet, QObject *parent=0);
	void removeGraphicsRep();// MZR 19082021



signals:
	void deletedMenu(); // MZR 19082021
private:
	double m_value;
	QString m_kind;

	QString m_markerName;
	QUuid m_uuid;

	WellBore* m_wellBore;
	Marker* m_currentMarker;
	WellPickGraphicRepFactory * m_repFactory;
};

#endif
