#include "wellboreproppanelon3d.h"

#include "wellbore.h"
#include "wellborerepon3d.h"

#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QFormLayout>
#include <QColorDialog>
#include <QLabel>

#include <limits>

WellBorePropPanelOn3D::WellBorePropPanelOn3D(WellBoreRepOn3D* rep,
		QWidget* parent) : QWidget(parent) {
	m_rep = rep;
	m_defaultWidth = 40;
	m_minimalWidth = 50;
	m_maximalWidth = 90;
	m_logMin = -1;
	m_logMax = 1;
	m_defaultColor = QColor(255, 255, 0);
	m_logIndex = m_rep->wellBore()->currentLogIndex();

	m_defaultWidthSave = m_defaultWidth;
	m_minimalWidthSave = m_minimalWidth;
	m_maximalWidthSave = m_maximalWidth;
	m_logMinSave = m_logMin;
	m_logMaxSave = m_logMax;
	m_defaultColorSave = m_defaultColor;
	m_logIndexSave = m_logIndex;

	buildWidget();

	updateLog();

	connect(m_rep->wellBore(), &WellBore::logChanged, this, &WellBorePropPanelOn3D::updateLog);

}

WellBorePropPanelOn3D::WellBorePropPanelOn3D(WellBoreRepOn3D* rep, long defaultWidth, long minimalWidth,
			long maximalWidth, double logMin, double logMax, QColor defaultColor, QWidget* parent) :
				QWidget(parent) {
	m_rep = rep;
	m_defaultWidth = defaultWidth;
	m_minimalWidth = minimalWidth;
	m_maximalWidth = maximalWidth;
	m_logMin = logMin;
	m_logMax = logMax;
	m_defaultColor = defaultColor;
	m_logIndex = m_rep->wellBore()->currentLogIndex();

	m_defaultWidthSave = m_defaultWidth;
	m_minimalWidthSave = m_minimalWidth;
	m_maximalWidthSave = m_maximalWidth;
	m_logMinSave = m_logMin;
	m_logMaxSave = m_logMax;
	m_defaultColorSave = m_defaultColor;
	m_logIndexSave = m_logIndex;

	buildWidget();

	updateLog();

	connect(m_rep->wellBore(), &WellBore::logChanged, this, &WellBorePropPanelOn3D::updateLog);
	connect(m_rep->wellBore(), &WellBore::boreUpdated, this, &WellBorePropPanelOn3D::updateWidget); // MZR 05082021
}

WellBorePropPanelOn3D::~WellBorePropPanelOn3D() {

}


void WellBorePropPanelOn3D::updateWidget() {
	m_logComboBox->clear();
	m_logComboBox->addItem("None", (qlonglong) m_noLogIndex);
	const std::vector<QString>& logsNames = m_rep->wellBore()->logsNames();
	for (qlonglong i=0; i<logsNames.size(); i++) {
		m_logComboBox->addItem(logsNames[i], i);
	}
}
/*
void WellBorePropPanelOn3D::computeMinMax()
{
	const Logs& log = m_rep->wellBore()->currentLog();
	bool isLogDefined = m_rep->wellBore()->isLogDefined() && log.unit==WellUnit::MD && log.nonNullIntervals.size()>0;

	double mini=  std::numeric_limits<double>::max();
	double maxi= std::numeric_limits<double>::lowest();


	if(isLogDefined)
	{
		for(int i=0;i< log.nonNullIntervals.size();i++)
		{
			int start = log.nonNullIntervals[i].first;
			int end = log.nonNullIntervals[i].second;
			for(int index=start;index<=end;index+= 1)
			{
				double valeur = log.attributes[index];
				if(valeur <mini ) mini = valeur;
				if(valeur > maxi) maxi = valeur;
			}
		}
	}

	setLogMin( mini);
	setLogMax( maxi);
	m_logMinSpinBox->setValue(m_logMin);
	m_logMaxSpinBox->setValue(m_logMax);

}*/

void WellBorePropPanelOn3D::updateLog() {
	setLog(m_rep->wellBore()->currentLogIndex());

	setLogMin( m_rep->wellBore()->mini());
	setLogMax(m_rep->wellBore()->maxi());
	m_logMinSpinBox->setValue(m_logMin);
	m_logMaxSpinBox->setValue(m_logMax);

}

void WellBorePropPanelOn3D::setLog(long index) {
	QSignalBlocker b1(m_logComboBox);

	// MZR 05082021
	if(index != -1){
		std::size_t indexComboBox=0;
		bool notFound = true;
		while(indexComboBox<m_logComboBox->count() && notFound) {
			bool ok;
			long searchedIndex = m_logComboBox->itemData(indexComboBox).toInt(&ok);
			notFound = !ok || index!=searchedIndex;
			if (notFound) {
				indexComboBox++;
			}
		}

		if (indexComboBox<m_logComboBox->count()) {
			m_logComboBox->setCurrentIndex(indexComboBox);
		}
		else{
			setLog(m_noLogIndex);
		}
	}
	return;
}

void WellBorePropPanelOn3D::setDefaultWidth(int defaultWidth) {
	QSignalBlocker b1(m_defaultWidthSpinBox);
	m_defaultWidthSpinBox->setValue(defaultWidth);
	m_defaultWidth = defaultWidth;
}

void WellBorePropPanelOn3D::setMinimalWidth(int minimalWidth) {
	QSignalBlocker b1(m_minimalWidthSpinBox);
	m_minimalWidthSpinBox->setValue(minimalWidth);
	m_minimalWidth = minimalWidth;
}

void WellBorePropPanelOn3D::setMaximalWidth(int maximalWidth) {
	QSignalBlocker b1(m_maximalWidthSpinBox);
	m_maximalWidthSpinBox->setValue(maximalWidth);
	m_maximalWidth = maximalWidth;
}

void WellBorePropPanelOn3D::setLogMin(double logMin) {
	QSignalBlocker b1(m_logMinSpinBox);
	m_logMinSpinBox->setValue(logMin);
	m_logMin = logMin;
}

void WellBorePropPanelOn3D::setLogMax(double logMax) {
	QSignalBlocker b1(m_logMaxSpinBox);
	m_logMaxSpinBox->setValue(logMax);
	m_logMax = logMax;
}

void WellBorePropPanelOn3D::setDefaultColor(QColor color) {
	m_defaultColor = color;
	m_defaultColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(color.name()));
}

void WellBorePropPanelOn3D::logChangedInternal(int index) {
	bool ok;
	long logIndex = m_logComboBox->itemData(index).toLongLong(&ok);
	if (logIndex==m_noLogIndex || !ok) {
		m_rep->wellBore()->selectLog(-1);
	} else {
		ok = m_rep->wellBore()->selectLog(logIndex);
		if (!ok) {
			setLog(m_noLogIndex);
		}
		/*else
		{
			computeMinMax();

		}*/
	}
	if (ok) {
		//emit logChanged(logIndex);
		m_logIndex = logIndex;
		checkCurrentState();
	}
}

void WellBorePropPanelOn3D::defaultWidthChangedInternal(int width) {
	if (width!=m_defaultWidth) {
		m_defaultWidth = width;
		checkCurrentState();
		//emit defaultWidthChanged(m_defaultWidth);
	}
}

void WellBorePropPanelOn3D::minimalWidthChangedInternal(int width) {
	if (width!=m_minimalWidth) {
		m_minimalWidth = width;
		checkCurrentState();
		//emit minimalWidthChanged(m_minimalWidth);
	}
}

void WellBorePropPanelOn3D::maximalWidthChangedInternal(int width) {
	if (width!=m_maximalWidth) {
		m_maximalWidth = width;
		checkCurrentState();
		//emit maximalWidthChanged(m_maximalWidth);
	}
}

void WellBorePropPanelOn3D::logMinChangedInternal(double logMin) {
	if (logMin!=m_logMin) {
		m_logMin = logMin;
		checkCurrentState();
		//emit logMinChanged(m_logMin);
	}
}

void WellBorePropPanelOn3D::logMaxChangedInternal(double logMax) {
	if (logMax!=m_logMax) {
		m_logMax = logMax;
		checkCurrentState();
		//emit logMaxChanged(m_logMax);
	}
}

void WellBorePropPanelOn3D::defaultColorChangedInternal() {
	QColor color = QColorDialog::getColor(m_defaultColor, this, tr("Default color chooser"));
	if (color != m_defaultColor) {
		m_defaultColor = color;
		m_defaultColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(color.name()));
		checkCurrentState();
		//emit defaultColorChanged(color);
	}
}

void WellBorePropPanelOn3D::applyChanges() {
	m_defaultWidthSave = m_defaultWidth;
	m_minimalWidthSave = m_minimalWidth;
	m_maximalWidthSave = m_maximalWidth;
	m_logMinSave = m_logMin;
	m_logMaxSave = m_logMax;
	m_defaultColorSave = m_defaultColor;
	m_logIndexSave = m_logIndex;

	checkCurrentState();
	emit stateUpdated();
}

void WellBorePropPanelOn3D::restoreState() {
	setLog(m_logIndexSave);
	setDefaultWidth(m_defaultWidthSave);
	setMinimalWidth(m_minimalWidthSave);
	setMaximalWidth(m_maximalWidthSave);
	setLogMin(m_logMinSave);
	setLogMax(m_logMaxSave);
	setDefaultColor(m_defaultColorSave);

	checkCurrentState();
}

void WellBorePropPanelOn3D::checkCurrentState() {
	if (m_logIndexSave==m_logIndex && m_defaultWidthSave==m_defaultWidth && m_minimalWidthSave==m_minimalWidth &&
			m_maximalWidthSave==m_maximalWidth && m_logMinSave==m_logMin && m_logMaxSave==m_logMax &&
			m_defaultColorSave==m_defaultColor) {
		m_restoreState->hide();
		m_applyChanges->hide();
	} else {
		m_restoreState->show();
		m_applyChanges->show();
	}
}

void WellBorePropPanelOn3D::buildWidget() {
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QFormLayout* layout = new QFormLayout;
	mainLayout->addLayout(layout);

	m_logComboBox = new QComboBox;
	m_noLogIndex = -1;
	m_logComboBox->addItem("None", (qlonglong) m_noLogIndex);
	const std::vector<QString>& logsNames = m_rep->wellBore()->logsNames();
	for (qlonglong i=0; i<logsNames.size(); i++) {
		m_logComboBox->addItem(logsNames[i], i);
	}
	layout->addRow("Log ", m_logComboBox);
	connect(m_logComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &WellBorePropPanelOn3D::logChangedInternal);

	m_defaultWidthSpinBox = new QSpinBox;
	m_defaultWidthSpinBox->setMinimum(1);
	m_defaultWidthSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_defaultWidthSpinBox->setValue(m_defaultWidth);
	layout->addRow("Default Width", m_defaultWidthSpinBox);
	connect(m_defaultWidthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this,
				&WellBorePropPanelOn3D::defaultWidthChangedInternal);

	m_minimalWidthSpinBox = new QSpinBox;
	m_minimalWidthSpinBox->setMinimum(1);
	m_minimalWidthSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_minimalWidthSpinBox->setValue(m_minimalWidth);
	layout->addRow("Minimal Width", m_minimalWidthSpinBox);
	connect(m_minimalWidthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this,
				&WellBorePropPanelOn3D::minimalWidthChangedInternal);

	m_maximalWidthSpinBox = new QSpinBox;
	m_maximalWidthSpinBox->setMinimum(1);
	m_maximalWidthSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_maximalWidthSpinBox->setValue(m_maximalWidth);
	layout->addRow("Maximal Width", m_maximalWidthSpinBox);
	connect(m_maximalWidthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this,
				&WellBorePropPanelOn3D::maximalWidthChangedInternal);

	m_logMinSpinBox = new QDoubleSpinBox;
	m_logMinSpinBox->setMinimum(std::numeric_limits<double>::lowest());
	m_logMinSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_logMinSpinBox->setValue(m_logMin);
	layout->addRow("Log min", m_logMinSpinBox);
	connect(m_logMinSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
				&WellBorePropPanelOn3D::logMinChangedInternal);

	m_logMaxSpinBox = new QDoubleSpinBox;
	m_logMaxSpinBox->setMinimum(std::numeric_limits<double>::lowest());
	m_logMaxSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_logMaxSpinBox->setValue(m_logMax);
	layout->addRow("Log max", m_logMaxSpinBox);
	connect(m_logMaxSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
				&WellBorePropPanelOn3D::logMaxChangedInternal);

	m_defaultColorHolder = new QPushButton;
	m_defaultColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_defaultColor.name()));
	m_defaultColorLabel = new QLabel("Fill top color");
	layout->addRow(m_defaultColorLabel, m_defaultColorHolder);
	connect(m_defaultColorHolder, &QPushButton::clicked, this,
			&WellBorePropPanelOn3D::defaultColorChangedInternal);

	m_applyChanges = new QPushButton("Apply changes");
	connect(m_applyChanges, &QPushButton::clicked, this,
				&WellBorePropPanelOn3D::applyChanges);
	mainLayout->addWidget(m_applyChanges);
	m_applyChanges->hide();

	m_restoreState = new QPushButton("Restore state");
	connect(m_restoreState, &QPushButton::clicked, this,
				&WellBorePropPanelOn3D::restoreState);
	mainLayout->addWidget(m_restoreState);
	m_restoreState->hide();
}

long WellBorePropPanelOn3D::defaultWidth() const {
	return m_defaultWidth;
}

long WellBorePropPanelOn3D::minimalWidth() const {
	return m_minimalWidth;
}

long WellBorePropPanelOn3D::maximalWidth() const {
	return m_maximalWidth;
}

double WellBorePropPanelOn3D::logMin() const {
	return m_logMin;
}

double WellBorePropPanelOn3D::logMax() const {
	return m_logMax;
}

QColor WellBorePropPanelOn3D::defaultColor() const {
	return m_defaultColor;
}
