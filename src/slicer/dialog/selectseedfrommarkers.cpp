#include "selectseedfrommarkers.h"
#include "workingsetmanager.h"
#include "marker.h"
#include "folderdata.h"
#include "seismic3dabstractdataset.h"

#include <QVBoxLayout>
#include <QListWidget>
#include <QListWidgetItem>
#include <QLabel>
#include <QDialogButtonBox>

SelectSeedFromMarkers::SelectSeedFromMarkers(WorkingSetManager* manager, Seismic3DAbstractDataset* dataset, int channel,
		SampleUnit sampleUnit, QString const& title, QWidget *parent, Qt::WindowFlags f) : QDialog(parent, f) {
	m_dataset = dataset;
	m_manager = manager;
	m_sampleUnit = sampleUnit;
	if (channel<0 || channel>=m_dataset->dimV()) {
		m_channel = 0;
	} else {
		m_channel = channel;
	}

	// get markers
	const QList<IData*>& datas = m_manager->folders().markers->data();
	for (IData* data : datas) {
		Marker* marker = dynamic_cast<Marker*>(data);
		if (marker!=nullptr) {
			m_markers.push_back(marker);
		}
	}

	QVBoxLayout* mainLayout = new QVBoxLayout();
	setLayout(mainLayout);

	QLabel* label = new QLabel("Markers");
	mainLayout->addWidget(label);

	// build list
	m_markersListWidget = new QListWidget;
	mainLayout->addWidget(m_markersListWidget);

	for (std::size_t i=0; i<m_markers.size(); i++) {
		Marker* marker = m_markers[i];
		QListWidgetItem* item = new QListWidgetItem(marker->name());

		m_markersListWidget->addItem(item);
	}
	m_markersListWidget->setSelectionMode(QAbstractItemView::MultiSelection);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	mainLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	connect(m_markersListWidget, &QListWidget::itemSelectionChanged, this, &SelectSeedFromMarkers::markerSelectionChanged);
}

SelectSeedFromMarkers::~SelectSeedFromMarkers() {

}

void SelectSeedFromMarkers::markerSelectionChanged() {
	QList<QListWidgetItem *> selection = m_markersListWidget->selectedItems();
	m_selectedMarkers.clear();

	for (QListWidgetItem* item : selection) {
		for (Marker* marker : m_markers) {
			if (item->text().compare(marker->name())==0) {
				m_selectedMarkers.push_back(marker);
			}
		}
	}
}

QList<RgtSeed> SelectSeedFromMarkers::getSelectedSeeds() const {
	QList<RgtSeed> out;
	for (Marker* marker : m_selectedMarkers) {
		QList<RgtSeed> markerList = marker->getProjectedPicksOnDataset(m_dataset, m_channel, m_sampleUnit);
		out.append(markerList);
	}
	return out;
}
