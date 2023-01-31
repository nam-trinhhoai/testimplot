#include "selectorcreatelayerdialog.h"
#include "rgblayerfromdataset.h"
#include "fixedlayerfromdataset.h"

#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QLineEdit>
#include <QSpinBox>


SelectOrCreateLayerDialog::SelectOrCreateLayerDialog(const QList<RgbLayerFromDataset*>& listRgb,
		const QList<FixedLayerFromDataset*>& listGray, QString defaultLabelPca, QString defaultLabelTmap) {
	m_listRgb = listRgb;
	m_listGray = listGray;
	m_defaultLabelTmap = defaultLabelTmap;
	m_defaultLabelPca = defaultLabelPca;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QGroupBox* processGroupBox = new QGroupBox("Process");
	QVBoxLayout* processLayout = new QVBoxLayout();
	processGroupBox->setLayout(processLayout);
	mainLayout->addWidget(processGroupBox);

	m_processComboBox = new QComboBox();
	processLayout->addWidget(m_processComboBox);
	m_processComboBox->addItem("Tmap", true);
	m_processComboBox->addItem("Pca", false);
	m_processComboBox->setCurrentIndex(0);
	m_chooseTmap = true;

	QGroupBox* layerGroupBox = new QGroupBox("Layer");
	QHBoxLayout* layerLayout = new QHBoxLayout;
	layerGroupBox->setLayout(layerLayout);
	mainLayout->addWidget(layerGroupBox);

	m_layerComboBox = new QComboBox;
	layerLayout->addWidget(m_layerComboBox);

	for (int index=0; index<m_listGray.size(); index++) {
		m_layerComboBox->addItem(m_listGray[index]->name(), index);
	}
	m_layerComboBox->addItem("New Layer", QVariant((int)-1));

	m_newLayerLineEdit = new QLineEdit;
	layerLayout->addWidget(m_newLayerLineEdit);

	QGroupBox* labelGroupBox = new QGroupBox("Label");
	QHBoxLayout* labelLayout = new QHBoxLayout;
	labelGroupBox->setLayout(labelLayout);
	mainLayout->addWidget(labelGroupBox);

	m_labelComboBox = new QComboBox;
	labelLayout->addWidget(m_labelComboBox);

	if (m_listGray.size()>0) {
		for (QString key : m_listGray[0]->keys()) {
			m_labelComboBox->addItem(key, 0);
		}
	}
	m_labelComboBox->addItem("New Label", QVariant((int)-1));

	m_newLabelLineEdit = new QLineEdit(m_defaultLabelTmap);
	m_label = m_defaultLabelTmap;
	labelLayout->addWidget(m_newLabelLineEdit);

	if (m_listGray.size()==0) {
		m_layerComboBox->hide();
		m_labelComboBox->hide();
		m_isLayerNew = true;
	} else {
		m_isLayerNew = false;
		m_layer = m_layerComboBox->currentText();
		bool ok;
		m_index = m_layerComboBox->currentData().toInt(&ok); // should be a valid index
		m_newLayerLineEdit->hide();
		if (m_listGray[0]->keys().size()>0) {
			m_newLabelLineEdit->hide();
			m_label = m_labelComboBox->currentText();
		}
	}

	QGroupBox* paramsGroupBox = new QGroupBox("Parameters");
	QFormLayout* paramsLayout = new QFormLayout;
	paramsGroupBox->setLayout(paramsLayout);
	mainLayout->addWidget(paramsGroupBox);

	m_tmapSizeSpinBox = new QSpinBox;
	m_tmapSizeSpinBox->setMinimum(1);
	m_tmapSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_tmapSizeSpinBox->setValue(m_paramTmapSize);

	paramsLayout->addRow("Tmap size", m_tmapSizeSpinBox);

	m_exampleStepSpinBox = new QSpinBox;
	m_exampleStepSpinBox->setMinimum(1);
	m_exampleStepSpinBox->setMaximum(m_maxExampleStep);
	m_exampleStepSpinBox->setValue(m_paramExampleStep);

	paramsLayout->addRow("Example extraction step", m_exampleStepSpinBox);


	QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal);
	mainLayout->addWidget(buttons);

	connect(buttons, &QDialogButtonBox::accepted, this, &SelectOrCreateLayerDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &SelectOrCreateLayerDialog::reject);

	connect(m_processComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
			&SelectOrCreateLayerDialog::changeProcess);

	connect(m_layerComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
			&SelectOrCreateLayerDialog::changeLayerIndex);

	connect(m_labelComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
		bool ok;
		int listIndex = m_labelComboBox->itemData(index).toInt(&ok);
		if (ok && listIndex>=0) {
			m_newLabelLineEdit->hide();
			m_label = m_labelComboBox->itemText(index);
		} else {
			m_newLabelLineEdit->show();
			QString defaultLabel;
			if (m_chooseTmap) {
				defaultLabel = m_defaultLabelTmap;
			} else {
				defaultLabel = m_defaultLabelPca;
			}
			m_newLabelLineEdit->setText(defaultLabel);
			m_label = defaultLabel;
		}
	});

	connect(m_newLayerLineEdit, &QLineEdit::editingFinished, this, &SelectOrCreateLayerDialog::newLayerLineEditChanged);
	connect(m_newLabelLineEdit, &QLineEdit::editingFinished, this, &SelectOrCreateLayerDialog::newLabelLineEditChanged);

	connect(m_tmapSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &SelectOrCreateLayerDialog::setParamTmapSize);
	connect(m_exampleStepSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &SelectOrCreateLayerDialog::setParamExampleStep);

}

SelectOrCreateLayerDialog::~SelectOrCreateLayerDialog() {

}

long SelectOrCreateLayerDialog::layerIndex() const {
	return m_index;
}

bool SelectOrCreateLayerDialog::isLayerNew() const {
	return m_isLayerNew;
}

QString SelectOrCreateLayerDialog::layer() const {
	return m_layer;
}

QString SelectOrCreateLayerDialog::label() const {
	return m_label;
}

void SelectOrCreateLayerDialog::newLayerLineEditChanged() {
	m_layer = m_newLayerLineEdit->text();
}

void SelectOrCreateLayerDialog::newLabelLineEditChanged() {
	m_label = m_newLabelLineEdit->text();
}


long SelectOrCreateLayerDialog::paramTmapSize() const {
	return m_paramTmapSize;
}

void SelectOrCreateLayerDialog::setParamTmapSize(long val) {
	m_paramTmapSize = val;
	m_tmapSizeSpinBox->setValue(m_paramTmapSize);
}

long SelectOrCreateLayerDialog::paramExampleStep() const {
	return m_paramExampleStep;
}

void SelectOrCreateLayerDialog::setParamExampleStep(long val) {
	m_paramExampleStep = val;
	m_exampleStepSpinBox->setValue(m_paramExampleStep);
}

long SelectOrCreateLayerDialog::maxExampleStep() const {
	return m_maxExampleStep;
}

void SelectOrCreateLayerDialog::setMaxExampleStep(long val) {
	m_maxExampleStep = val;
	m_exampleStepSpinBox->setMaximum(m_maxExampleStep);
}

bool SelectOrCreateLayerDialog::isTmapChoosen() const {
	return m_chooseTmap;
}

bool SelectOrCreateLayerDialog::isPcaChoosen() const {
	return !m_chooseTmap;
}

void SelectOrCreateLayerDialog::changeLayerIndex(int index) {
	if (m_chooseTmap) {
		bool ok;
		int listIndex = m_layerComboBox->itemData(index).toInt(&ok);
		if (ok && listIndex>=0) {
			m_newLayerLineEdit->hide();
			m_layer = m_layerComboBox->itemText(index);
			m_index = listIndex;
			m_isLayerNew = false;

			m_labelComboBox->clear();
			for (QString key : m_listGray[index]->keys()) {
				m_labelComboBox->addItem(key, 0);
			}
			m_labelComboBox->addItem("New Label", QVariant((int)-1));
			m_labelComboBox->show();
//			m_newLabelLineEdit->hide();
			// no need to set label value
		} else {
			m_newLayerLineEdit->show();
			m_layer = m_newLayerLineEdit->text();
			m_index = -1;
			m_isLayerNew = true;

			m_labelComboBox->clear();
			m_labelComboBox->addItem("New Label", QVariant((int)-1));
			m_labelComboBox->hide();
//			m_newLabelLineEdit->show();
//			m_label = m_newLabelLineEdit->text();
		}
	} else {
		bool ok;
		int listIndex = m_layerComboBox->itemData(index).toInt(&ok);
		if (ok && listIndex>=0) {
			m_newLayerLineEdit->hide();
			m_layer = m_layerComboBox->itemText(index);
			m_index = listIndex;
			m_isLayerNew = false;

			m_labelComboBox->clear();
			for (QString key : m_listRgb[index]->keys()) {
				//get prefix because pca add number at the end of the prefix
				long index = key.size()-1;
				QString prefix;
				while (index>=0 && key.at(index).isDigit()) {
					index--;
				}
				if (index<0) {
					prefix = "";
				} else {
					prefix = key.chopped(key.size()-1 - index);
				}
				if (!prefix.isNull()) {
					long indexComboBox = m_labelComboBox->count()-1;
					while(indexComboBox>=0 && prefix.compare(m_labelComboBox->itemText(indexComboBox))!=0) {
						indexComboBox--;
					}
					if (indexComboBox<0) {
						m_labelComboBox->addItem(prefix, 0);
					}
				}
			}
			m_labelComboBox->addItem("New Label", QVariant((int)-1));
			m_labelComboBox->show();
//			m_newLabelLineEdit->hide();
			// no need to set label value
		} else {
			m_newLayerLineEdit->show();
			m_layer = m_newLayerLineEdit->text();
			m_index = -1;
			m_isLayerNew = true;

			m_labelComboBox->clear();
			m_labelComboBox->addItem("New Label", QVariant((int)-1));
			m_labelComboBox->hide();
//			m_newLabelLineEdit->show();
//			m_label = m_newLabelLineEdit->text();
		}

	}
}

void SelectOrCreateLayerDialog::changeProcess(int index) {
	if (index>=0 && index<m_processComboBox->count()) {
		bool newVal = m_processComboBox->itemData(index).toBool();
		if (newVal!=m_chooseTmap) {
			m_chooseTmap = newVal;
			m_layerComboBox->clear();
			if (m_chooseTmap) {
				// no need to clear labels because it should react when first item added
				for (int index=0; index<m_listGray.size(); index++) {
					m_layerComboBox->addItem(m_listGray[index]->name(), index);
				}
				m_layerComboBox->addItem("New Layer", QVariant((int)-1));
			} else {
				// no need to clear labels because it should react when first item added
				for (int index=0; index<m_listRgb.size(); index++) {
					m_layerComboBox->addItem(m_listRgb[index]->name(), index);
				}
				m_layerComboBox->addItem("New Layer", QVariant((int)-1));
			}
			if (m_layerComboBox->count()==1) {
				m_layerComboBox->hide();
			} else {
				m_layerComboBox->show();
			}
		}
	}
}
