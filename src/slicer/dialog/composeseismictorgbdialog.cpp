#include "composeseismictorgbdialog.h"
#include "seismic3dabstractdataset.h"
#include "workingsetmanager.h"
#include "folderdata.h"
#include "seismicsurvey.h"

#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QComboBox>
#include <QSpinBox>

ComposeSeismicToRgbDialog::ComposeSeismicToRgbDialog(WorkingSetManager* workingSet) {
	const QList<IData*>& surveyDatas = workingSet->folders().seismics->data();
	for (IData* surveyData : surveyDatas) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyData)) {
			for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
				m_allDatasets.append(dataset);
			}
		}
	}

	initObject();
}

ComposeSeismicToRgbDialog::ComposeSeismicToRgbDialog(SeismicSurvey* survey) {
	for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
		m_allDatasets.append(dataset);
	}

	initObject();
}

void ComposeSeismicToRgbDialog::initObject() {
	m_red = nullptr;
	m_channelRed = 0;
	m_green = nullptr;
	m_channelGreen = 0;
	m_blue = nullptr;
	m_channelBlue = 0;
	m_alpha = nullptr;
	m_channelAlpha = 0;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QGridLayout* gridLayout = new QGridLayout;
	mainLayout->addLayout(gridLayout);

	m_comboBoxRed = new QComboBox;
	m_spinBoxRed = new QSpinBox;
	setupChannel(QString("Red"), m_comboBoxRed, m_spinBoxRed, gridLayout, 0);

	m_comboBoxGreen = new QComboBox;
	m_spinBoxGreen = new QSpinBox;
	setupChannel(QString("Green"), m_comboBoxGreen, m_spinBoxGreen, gridLayout, 1);

	m_comboBoxBlue = new QComboBox;
	m_spinBoxBlue = new QSpinBox;
	setupChannel(QString("Blue"), m_comboBoxBlue, m_spinBoxBlue, gridLayout, 2);

	m_comboBoxAlpha = new QComboBox;
	m_spinBoxAlpha = new QSpinBox;
	m_comboBoxAlpha->addItem("None", -1);
	setupChannel(QString("Alpha"), m_comboBoxAlpha, m_spinBoxAlpha, gridLayout, 3);

	if (m_allDatasets.size()>0) {
		m_red = m_allDatasets[0];
		m_green = m_allDatasets[0];
		m_blue = m_allDatasets[0];
	}

	QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal);
	mainLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &ComposeSeismicToRgbDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &ComposeSeismicToRgbDialog::reject);

	connect(m_comboBoxRed, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ComposeSeismicToRgbDialog::changeRed);
	connect(m_comboBoxGreen, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ComposeSeismicToRgbDialog::changeGreen);
	connect(m_comboBoxBlue, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ComposeSeismicToRgbDialog::changeBlue);
	connect(m_comboBoxAlpha, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ComposeSeismicToRgbDialog::changeAlpha);

	connect(m_spinBoxRed, QOverload<int>::of(&QSpinBox::valueChanged), this, &ComposeSeismicToRgbDialog::changeRedChannel);
	connect(m_spinBoxGreen, QOverload<int>::of(&QSpinBox::valueChanged), this, &ComposeSeismicToRgbDialog::changeGreenChannel);
	connect(m_spinBoxBlue, QOverload<int>::of(&QSpinBox::valueChanged), this, &ComposeSeismicToRgbDialog::changeBlueChannel);
	connect(m_spinBoxAlpha, QOverload<int>::of(&QSpinBox::valueChanged), this, &ComposeSeismicToRgbDialog::changeAlphaChannel);
}

ComposeSeismicToRgbDialog::~ComposeSeismicToRgbDialog() {

}

Seismic3DAbstractDataset* ComposeSeismicToRgbDialog::red() const {
	return m_red;
}

int ComposeSeismicToRgbDialog::channelRed() const {
	return m_channelRed;
}

Seismic3DAbstractDataset* ComposeSeismicToRgbDialog::green() const {
	return m_green;
}

int ComposeSeismicToRgbDialog::channelGreen() const {
	return m_channelGreen;
}

Seismic3DAbstractDataset* ComposeSeismicToRgbDialog::blue() const {
	return m_blue;
}

int ComposeSeismicToRgbDialog::channelBlue() const {
	return m_channelBlue;
}

Seismic3DAbstractDataset* ComposeSeismicToRgbDialog::alpha() const {
	return m_alpha;
}

int ComposeSeismicToRgbDialog::channelAlpha() const {
	return m_channelAlpha;
}

void ComposeSeismicToRgbDialog::fillComboBox(QComboBox* comboBox) {
	for (std::size_t i=0; i< m_allDatasets.size(); i++) {
		Seismic3DAbstractDataset* dataset = m_allDatasets[i];
		comboBox->addItem(dataset->name(), QVariant((int)i));
	}
}

void ComposeSeismicToRgbDialog::setupChannel(const QString& name, QComboBox* comboBox, QSpinBox* spinBox, QGridLayout* gridLayout, int row) {
	QLabel* labelRed = new QLabel(name);
	gridLayout->addWidget(labelRed, row, 0);

	fillComboBox(comboBox);
	gridLayout->addWidget(comboBox, row, 1);

	spinBox->setMinimum(0);
	if (m_allDatasets.size()>0) {
		spinBox->setMaximum(m_allDatasets[0]->dimV()-1);
	} else {
		spinBox->setMaximum(0);
	}
	spinBox->setValue(0);
	gridLayout->addWidget(spinBox, row, 2);

	if (spinBox->maximum()==0) {
		spinBox->hide();
	}
}

void ComposeSeismicToRgbDialog::changeRed(int index) {
	bool ok;
	int listIndex = m_comboBoxRed->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		m_red = m_allDatasets[listIndex];
		m_spinBoxRed->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		m_red = nullptr;
		m_spinBoxRed->setMaximum(0);
	}

	if (m_spinBoxRed->maximum()>0) {
		m_spinBoxRed->show();
	} else {
		m_spinBoxRed->hide();
	}
}

void ComposeSeismicToRgbDialog::changeGreen(int index) {
	bool ok;
	int listIndex = m_comboBoxGreen->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		m_green = m_allDatasets[listIndex];
		m_spinBoxGreen->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		m_green = nullptr;
		m_spinBoxGreen->setMaximum(0);
	}

	if (m_spinBoxGreen->maximum()>0) {
		m_spinBoxGreen->show();
	} else {
		m_spinBoxGreen->hide();
	}
}

void ComposeSeismicToRgbDialog::changeBlue(int index) {
	bool ok;
	int listIndex = m_comboBoxBlue->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		m_blue = m_allDatasets[listIndex];
		m_spinBoxBlue->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		m_blue = nullptr;
		m_spinBoxBlue->setMaximum(0);
	}

	if (m_spinBoxBlue->maximum()>0) {
		m_spinBoxBlue->show();
	} else {
		m_spinBoxBlue->hide();
	}
}

void ComposeSeismicToRgbDialog::changeAlpha(int index) {
	bool ok;
	int listIndex = m_comboBoxAlpha->itemData(index).toInt(&ok);
	if (ok && listIndex>=0) {
		m_alpha = m_allDatasets[listIndex];
		m_spinBoxAlpha->setMaximum(m_allDatasets[listIndex]->dimV()-1);
	} else {
		m_alpha = nullptr;
		m_spinBoxAlpha->setMaximum(0);
	}

	if (m_spinBoxAlpha->maximum()>0) {
		m_spinBoxAlpha->show();
	} else {
		m_spinBoxAlpha->hide();
	}
}

void ComposeSeismicToRgbDialog::changeRedChannel(int channel) {
	m_channelRed = channel;
}

void ComposeSeismicToRgbDialog::changeGreenChannel(int channel) {
	m_channelGreen = channel;
}

void ComposeSeismicToRgbDialog::changeBlueChannel(int channel) {
	m_channelBlue = channel;
}

void ComposeSeismicToRgbDialog::changeAlphaChannel(int channel) {
	m_channelAlpha = channel;
}
