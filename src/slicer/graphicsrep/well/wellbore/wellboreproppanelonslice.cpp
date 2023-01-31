#include "wellboreproppanelonslice.h"

#include "wellbore.h"

#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QFormLayout>
#include <QColorDialog>
#include <QLabel>
#include <QDebug>
#include <QMessageBox>

#include <limits>

#include <loginformationaggregator.h>
#include <managerFileSelectorWidget.h>

WellBorePropPanelOnSlice::WellBorePropPanelOnSlice(WellBore* data,
		QWidget* parent) : QWidget(parent) {
	m_data = data;
	m_origin = 0;
	m_width = 100;
	m_logMin = -1;
	m_logMax = 1;
	m_base = false;
	m_logBase = 0;
	m_fillTop = false;
	m_fillTopColor = QColor(Qt::blue);
	m_fillBottom = false;
	m_fillBottomColor = QColor(Qt::red);

	buildWidget();

	updateLog();

	connect(m_data, &WellBore::logColorChanged, this, &WellBorePropPanelOnSlice::setColor);
	connect(m_data, &WellBore::logChanged, this, &WellBorePropPanelOnSlice::updateLog);
	connect(m_data, &WellBore::boreUpdated, this, &WellBorePropPanelOnSlice::updateWidget); // MZR 04082021

}

WellBorePropPanelOnSlice::~WellBorePropPanelOnSlice() {

}

void WellBorePropPanelOnSlice::updateWidget() {
	m_logComboBox->clear();
	m_logComboBox->addItem("None", (qlonglong) m_noLogIndex);
	const std::vector<QString>& logsNames = m_data->logsNames();
	for (qlonglong i=0; i<logsNames.size(); i++) {
		m_logComboBox->addItem(logsNames[i], i);
	}
}

void WellBorePropPanelOnSlice::updateLog() {
	setLog(m_data->currentLogIndex());

	if(m_data->mini() != m_logMin){
	   setLogMin( m_data->mini());
	   logMinChangedInternal(m_logMin);
	}
	if(m_data->maxi() != m_logMax){
	   setLogMax(m_data->maxi());
	   logMinChangedInternal(m_logMax);
	}
}

void WellBorePropPanelOnSlice::setLog(long index) {
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

		if (indexComboBox<m_logComboBox->count()){
			m_logComboBox->setCurrentIndex(indexComboBox);
		}
		else{
			setLog(m_noLogIndex);
		}
	}
	return;
}

void WellBorePropPanelOnSlice::setColor(QColor color) {
	m_colorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_data->logColor().name()));
}

void WellBorePropPanelOnSlice::setOrigin(int origin) {
	QSignalBlocker b1(m_originSpinBox);
	m_originSpinBox->setValue(origin);
	m_origin = origin;
}

void WellBorePropPanelOnSlice::setWidth(int width) {
	QSignalBlocker b1(m_widthSpinBox);
	m_widthSpinBox->setValue(width);
	m_width = width;
}

void WellBorePropPanelOnSlice::setLogMin(double logMin) {
	QSignalBlocker b1(m_logMinSpinBox);
	m_logMinSpinBox->setValue(logMin);
	m_logMin = logMin;
}

void WellBorePropPanelOnSlice::setLogMax(double logMax) {
	QSignalBlocker b1(m_logMaxSpinBox);
	m_logMaxSpinBox->setValue(logMax);
	m_logMax = logMax;
}

void WellBorePropPanelOnSlice::setBase(bool activated) {
	QSignalBlocker b1(m_baseCheckBox);
	m_baseCheckBox->setCheckState((activated) ? Qt::Checked : Qt::Unchecked);
	m_base = activated;
	if (m_base) {
		m_logBaseSpinBox->show();
		m_logBaseLabel->show();
	} else {
		m_logBaseSpinBox->hide();
		m_logBaseLabel->hide();
	}
}

void WellBorePropPanelOnSlice::setLogBase(double base) {
	QSignalBlocker b1(m_logBaseSpinBox);
	m_logBaseSpinBox->setValue(base);
	m_logBase = base;
}

void WellBorePropPanelOnSlice::setFillTop(bool activated) {
	QSignalBlocker b1(m_fillTopCheckBox);
	m_fillTopCheckBox->setCheckState((activated) ? Qt::Checked : Qt::Unchecked);
	m_fillTop = activated;
	if (m_fillTop) {
		m_fillTopColorHolder->show();
		m_fillTopColorLabel->show();
	} else {
		m_fillTopColorHolder->hide();
		m_fillTopColorLabel->hide();
	}
}

void WellBorePropPanelOnSlice::setFillTopColor(QColor color) {
	m_fillTopColor = color;
	m_fillTopColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(color.name()));
}

void WellBorePropPanelOnSlice::setFillBottom(bool activated) {
	QSignalBlocker b1(m_fillBottomCheckBox);
	m_fillBottomCheckBox->setCheckState((activated) ? Qt::Checked : Qt::Unchecked);
	m_fillBottom = activated;
	if (m_fillBottom) {
		m_fillBottomColorHolder->show();
		m_fillBottomColorLabel->show();
	} else {
		m_fillBottomColorHolder->hide();
		m_fillBottomColorLabel->hide();
	}
}

void WellBorePropPanelOnSlice::setFillBottomColor(QColor color) {
	m_fillBottomColor = color;
	m_fillBottomColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(color.name()));
}

void WellBorePropPanelOnSlice::logChangedInternal(int index) {
	bool ok;
	long logIndex = m_logComboBox->itemData(index).toLongLong(&ok);
	// if ( m_bpLog ) m_bpLog->setText(m_logComboBox->currentText());
	if (logIndex==m_noLogIndex || !ok) {
		m_data->selectLog(-1);
	} else {
		ok = m_data->selectLog(logIndex);
		if (!ok) {
			setLog(m_noLogIndex);
		}
	}
	if (ok) {
		emit logChanged(logIndex);
	}
}

void WellBorePropPanelOnSlice::colorChangedInternal() {
	QColor color = QColorDialog::getColor(m_data->logColor(), this, tr("Log color chooser"));
	if (color != m_data->logColor()) {
		m_data->setLogColor(color);
		emit colorChanged(color);
	}
}

void WellBorePropPanelOnSlice::originChangedInternal(int origin) {
	if (origin!=m_origin) {
		m_origin = origin;
		emit originChanged(m_origin);
	}
}

void WellBorePropPanelOnSlice::widthChangedInternal(int width) {
	if (width!=m_width) {
		m_width = width;
		emit widthChanged(m_width);
	}
}

void WellBorePropPanelOnSlice::logMinChangedInternal(double logMin) {
	if (logMin!=m_logMin) {
		m_logMin = logMin;
		emit logMinChanged(m_logMin);
	}
}

void WellBorePropPanelOnSlice::logMaxChangedInternal(double logMax) {
	if (logMax!=m_logMax) {
		m_logMax = logMax;
		emit logMaxChanged(m_logMax);
	}
}

void WellBorePropPanelOnSlice::baseChangedInternal(int state) {
	bool activated = state == Qt::Checked;
	if (activated!=m_base) {
		m_base = activated;
		if (m_base) {
			m_logBaseSpinBox->show();
			m_logBaseLabel->show();
		} else {
			m_logBaseSpinBox->hide();
			m_logBaseLabel->hide();
		}
		emit baseChanged(m_base);
	}
}

void WellBorePropPanelOnSlice::logBaseChangedInternal(double base) {
	if (base!=m_logBase) {
		m_logBase = base;
		emit logBaseChanged(m_logBase);
	}
}

void WellBorePropPanelOnSlice::fillTopChangedInternal(int state) {
	bool activated = state == Qt::Checked;
	if (activated!=m_fillTop) {
		m_fillTop = activated;
		if (m_fillTop) {
			m_fillTopColorHolder->show();
			m_fillTopColorLabel->show();
		} else {
			m_fillTopColorHolder->hide();
			m_fillTopColorLabel->hide();
		}
		emit fillTopChanged(m_fillTop);
	}
}

void WellBorePropPanelOnSlice::fillTopColorChangedInternal() {
	QColor color = QColorDialog::getColor(m_fillTopColor, this, tr("Fill top color chooser"));
	if (color != m_fillTopColor) {
		m_fillTopColor = color;
		emit fillTopColorChanged(color);
	}
}

void WellBorePropPanelOnSlice::fillBottomChangedInternal(int state) {
	bool activated = state == Qt::Checked;
	if (activated!=m_fillBottom) {
		m_fillBottom = activated;
		if (m_fillBottom) {
			m_fillBottomColorHolder->show();
			m_fillBottomColorLabel->show();
		} else {
			m_fillBottomColorHolder->hide();
			m_fillBottomColorLabel->hide();
		}
		emit fillBottomChanged(m_fillBottom);
	}
}

void WellBorePropPanelOnSlice::fillBottomColorChangedInternal() {
	QColor color = QColorDialog::getColor(m_fillBottomColor, this, tr("Fill bottom color chooser"));
	if (color != m_fillBottomColor) {
		m_fillBottomColor = color;
		emit fillBottomColorChanged(color);
	}
}

void WellBorePropPanelOnSlice::buildWidget() {
	QFormLayout* layout = new QFormLayout;
	setLayout(layout);

	// m_logLabel = new QLabel("test");
	// layout->addRow("Log ", m_logLabel);
	// connect(m_logLabel, SIGNAL(clicked()), this, SLOT(logLabelClick)); //&WellBorePropPanelOnSlice::logLabelClick);

	m_bpLog = new QPushButton("None");
	layout->addRow("Log ", m_bpLog);
	connect(m_bpLog, &QPushButton::clicked, this, &WellBorePropPanelOnSlice::logLabelClick);

	m_logComboBox = new QComboBox;
	m_noLogIndex = -1;
	m_logComboBox->addItem("None", (qlonglong) m_noLogIndex);
	const std::vector<QString>& logsNames = m_data->logsNames();
	for (qlonglong i=0; i<logsNames.size(); i++) {
		m_logComboBox->addItem(logsNames[i], i);
	}
	// layout->addRow("Log ", m_logComboBox);
	connect(m_logComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &WellBorePropPanelOnSlice::logChangedInternal);

	m_colorHolder = new QPushButton;
	m_colorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_data->logColor().name()));
	layout->addRow("Log color", m_colorHolder);
	connect(m_colorHolder, &QPushButton::clicked, this,
			&WellBorePropPanelOnSlice::colorChangedInternal);

	m_originSpinBox = new QSpinBox;
	m_originSpinBox->setMinimum(std::numeric_limits<int>::min());
	m_originSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_originSpinBox->setValue(m_origin);
	layout->addRow("Origin", m_originSpinBox);
	connect(m_originSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this,
			&WellBorePropPanelOnSlice::originChangedInternal);

	m_widthSpinBox = new QSpinBox;
	m_widthSpinBox->setMinimum(1);
	m_widthSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_widthSpinBox->setValue(m_width);
	layout->addRow("Width", m_widthSpinBox);
	connect(m_widthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this,
			&WellBorePropPanelOnSlice::widthChangedInternal);

	m_logMinSpinBox = new QDoubleSpinBox;
	m_logMinSpinBox->setMinimum(std::numeric_limits<double>::lowest());
	m_logMinSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_logMinSpinBox->setValue(m_logMin);
	layout->addRow("Log min", m_logMinSpinBox);
	connect(m_logMinSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
			&WellBorePropPanelOnSlice::logMinChangedInternal);

	m_logMaxSpinBox = new QDoubleSpinBox;
	m_logMaxSpinBox->setMinimum(std::numeric_limits<double>::lowest());
	m_logMaxSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_logMaxSpinBox->setValue(m_logMax);
	layout->addRow("Log max", m_logMaxSpinBox);
	connect(m_logMaxSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
			&WellBorePropPanelOnSlice::logMaxChangedInternal);

	m_baseCheckBox = new QCheckBox;
	m_baseCheckBox->setCheckState((m_base) ? Qt::Checked : Qt::Unchecked);
	layout->addRow("Base", m_baseCheckBox);
	connect(m_baseCheckBox, &QCheckBox::stateChanged, this,
			&WellBorePropPanelOnSlice::baseChangedInternal);

	m_logBaseSpinBox = new QDoubleSpinBox;
	m_logBaseSpinBox->setMinimum(std::numeric_limits<double>::lowest());
	m_logBaseSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_logBaseSpinBox->setValue(m_logBase);
	m_logBaseLabel = new QLabel("Log base");
	layout->addRow(m_logBaseLabel, m_logBaseSpinBox);
	connect(m_logBaseSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
			&WellBorePropPanelOnSlice::logBaseChangedInternal);

	if (!m_base) {
		m_logBaseSpinBox->hide();
		m_logBaseLabel->hide();
	}

	m_fillTopCheckBox = new QCheckBox;
	m_fillTopCheckBox->setCheckState((m_fillTop) ? Qt::Checked : Qt::Unchecked);
	layout->addRow("Fill top", m_fillTopCheckBox);
	connect(m_fillTopCheckBox, &QCheckBox::stateChanged, this,
			&WellBorePropPanelOnSlice::fillTopChangedInternal);

	m_fillTopColorHolder = new QPushButton;
	m_fillTopColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_fillTopColor.name()));
	m_fillTopColorLabel = new QLabel("Fill top color");
	layout->addRow(m_fillTopColorLabel, m_fillTopColorHolder);
	connect(m_fillTopColorHolder, &QPushButton::clicked, this,
			&WellBorePropPanelOnSlice::fillTopColorChangedInternal);

	if (!m_fillTop) {
		m_fillTopColorHolder->hide();
		m_fillTopColorLabel->hide();
	}

	m_fillBottomCheckBox = new QCheckBox;
	m_fillBottomCheckBox->setCheckState((m_fillBottom) ? Qt::Checked : Qt::Unchecked);
	layout->addRow("Fill bottom", m_fillBottomCheckBox);
	connect(m_fillBottomCheckBox, &QCheckBox::stateChanged, this,
			&WellBorePropPanelOnSlice::fillBottomChangedInternal);

	m_fillBottomColorHolder = new QPushButton;
	m_fillBottomColorHolder->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_fillBottomColor.name()));
	m_fillBottomColorLabel = new QLabel("Fill bottom color");
	layout->addRow(m_fillBottomColorLabel, m_fillBottomColorHolder);
	connect(m_fillBottomColorHolder, &QPushButton::clicked, this,
			&WellBorePropPanelOnSlice::fillBottomColorChangedInternal);

	if (!m_fillBottom) {
		m_fillBottomColorHolder->hide();
		m_fillBottomColorLabel->hide();
	}
}

long WellBorePropPanelOnSlice::origin() const {
	return m_origin;
}

long WellBorePropPanelOnSlice::width() const {
	return m_width;
}

double WellBorePropPanelOnSlice::logMin() const {
	return m_logMin;
}

double WellBorePropPanelOnSlice::logMax() const {
	return m_logMax;
}

bool WellBorePropPanelOnSlice::base() const {
	return m_base;
}

double WellBorePropPanelOnSlice::logBase() const {
	return m_logBase;
}

bool WellBorePropPanelOnSlice::fillTop() const {
	return m_fillTop;
}

QColor WellBorePropPanelOnSlice::fillTopColor() const {
	return m_fillTopColor;
}

bool WellBorePropPanelOnSlice::fillBottom() const {
	return m_fillBottom;
}

QColor WellBorePropPanelOnSlice::fillBottomColor() const {
	return m_fillBottomColor;
}


int WellBorePropPanelOnSlice::getLogNameIndex(QString name)
{
	const std::vector<QString>& logsNames = m_data->logsNames();
	for (int i=0; i<logsNames.size(); i++)
	{
		if ( name == logsNames[i] ) return i;
	}
	return -1;
}

void WellBorePropPanelOnSlice::logLabelClick()
{
	const std::vector<QString>& logsNames = m_data->logsNames();
	const std::vector<QString>& logsPath = m_data->logsFiles();
//	for (int i=0; i<logsNames.size(); i++)
//		qDebug() << logsNames[i] << "   -    " <<  logsPath[i];

	LogInformationAggregator* aggregator = new LogInformationAggregator(m_data);
	ManagerFileSelectorWidget* widget = new ManagerFileSelectorWidget(aggregator);
	int code = widget->exec();
    if (code==QDialog::Accepted)
    {
    	std::pair<std::vector<QString>, std::vector<QString>> names = widget->getSelectedNames();
    	if ( names.first.size() > 0 )
    	{
    		m_currentLogName = names.first[0];
    		m_currentLogPath = names.second[0];
    		int idx =getLogNameIndex(m_currentLogName);
    		m_bpLog->setText(m_currentLogName);
    		logChangedInternal(idx+1);
    	}
    }
	delete widget;
}
