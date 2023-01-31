#include "fixedlayerfromdatasetproppanelonslice.h"
#include "fixedlayerfromdatasetreponslice.h"
#include "fixedlayerfromdataset.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QColorDialog>

FixedLayerFromDatasetPropPanelOnSlice::FixedLayerFromDatasetPropPanelOnSlice(
		FixedLayerFromDatasetRepOnSlice *rep, QWidget *parent) : QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	QWidget* colorHolder = new QWidget;
	QLabel* colorLabel = new QLabel("Pick color : ");
	m_colorButton = new QPushButton;
	QHBoxLayout* layout = new QHBoxLayout;
	colorHolder->setLayout(layout);
	layout->addWidget(colorLabel);
	layout->addWidget(m_colorButton);

	processLayout->addWidget(colorHolder, 0, Qt::AlignmentFlag::AlignTop);
	m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_rep->fixedLayer()->getColor().name()));

	connect(m_colorButton, &QPushButton::clicked, this, &FixedLayerFromDatasetPropPanelOnSlice::selectColor);
	connect(m_rep->fixedLayer(), &FixedLayerFromDataset::colorChanged, this, &FixedLayerFromDatasetPropPanelOnSlice::setColor);
}

FixedLayerFromDatasetPropPanelOnSlice::~FixedLayerFromDatasetPropPanelOnSlice() {
	disconnect(m_rep->fixedLayer(), &FixedLayerFromDataset::colorChanged, this, &FixedLayerFromDatasetPropPanelOnSlice::setColor);
}

void FixedLayerFromDatasetPropPanelOnSlice::setColor(QColor color) {
	m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(color.name()));
	if (m_rep->fixedLayer()->getColor()!=color) {
		m_rep->fixedLayer()->setColor(color);
	}
}

void FixedLayerFromDatasetPropPanelOnSlice::selectColor() {
	QColorDialog dialog;
	dialog.setCurrentColor(m_rep->fixedLayer()->getColor());
	int result = dialog.exec();
	if (result==QDialog::Accepted) {
		setColor(dialog.currentColor());
	}
}
