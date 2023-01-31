#ifndef SelectSeedFromMarkers_H_
#define SelectSeedFromMarkers_H_

#include "RgtLayerProcessUtil.h"
#include "viewutils.h"

#include <QDialog>
#include <QString>
#include <QList>

class Marker;
class WorkingSetManager;
class Seismic3DAbstractDataset;
class QListWidget;

class SelectSeedFromMarkers :public QDialog {
Q_OBJECT
public:
	SelectSeedFromMarkers(WorkingSetManager* manager, Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit, QString const& title,
			QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	virtual ~SelectSeedFromMarkers();

	QList<RgtSeed> getSelectedSeeds() const;

private slots:
	void markerSelectionChanged();

private:
	QListWidget* m_markersListWidget;

	QList<Marker*> m_selectedMarkers;
	QList<Marker*> m_markers;
	WorkingSetManager* m_manager;
	Seismic3DAbstractDataset* m_dataset;
	int m_channel;
	SampleUnit m_sampleUnit;
};

#endif
