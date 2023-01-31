#include "exportnvhorizondialog.h"

#include "fixedlayerimplfreehorizonfromdatasetandcube.h"
#include "freehorizon.h"
#include "freeHorizonManager.h"
#include "isochron.h"
#include "isochronattribut.h"
#include "nextvisiondbmanager.h"
#include "nvlineedit.h"
#include "seismicDatabaseManager.h"
#include "seismicsurvey.h"
#include "sismagedbmanager.h"
#include "viewutils.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QSizeGrip>
#include <QVBoxLayout>


ExportNVHorizonDialog::ExportNVHorizonDialog(FreeHorizon* freeHorizon, QWidget* parent) :
		QDialog(parent), m_freeHorizon(freeHorizon) {
	connect(m_freeHorizon, &QObject::destroyed, this, &ExportNVHorizonDialog::horizonDestroyed);

	setWindowTitle("Export : " + freeHorizon->name());

	m_surveyPath = m_freeHorizon->survey()->idPath().toStdString();
	std::string sismageHorizonPath = SismageDBManager::surveyPath2HorizonsPath(m_surveyPath);
	m_sismageHorizonPath = QString::fromStdString(sismageHorizonPath);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);
	mainLayout->setContentsMargins(0, 0, 0, 0);

	QWidget* coreWidget = new QWidget;
	QVBoxLayout* coreLayout = new QVBoxLayout;
	coreWidget->setLayout(coreLayout);
	mainLayout->addWidget(coreWidget);

	QHBoxLayout* listLayout = new QHBoxLayout;
	coreLayout->addLayout(listLayout);

	m_attributsListWidget = createAttributsList();
	listLayout->addWidget(m_attributsListWidget);

	QVBoxLayout* saveHorizonLayout = new QVBoxLayout;
	listLayout->addLayout(saveHorizonLayout);

	QHBoxLayout* newLayout = new QHBoxLayout;
	saveHorizonLayout->addLayout(newLayout);
	newLayout->addWidget(new QLabel("New horizon"));
	m_newNameLineEdit = new NvLineEdit;
	newLayout->addWidget(m_newNameLineEdit);

	m_targetHorizonList = createTargetHorizonList();
	saveHorizonLayout->addWidget(m_targetHorizonList);

	QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	coreLayout->addWidget(buttonBox);

	mainLayout->addWidget(new QSizeGrip(this), 0, Qt::AlignRight);

	connect(m_targetHorizonList, &QListWidget::currentItemChanged,
			this, &ExportNVHorizonDialog::slotSelectItem);
	connect(m_newNameLineEdit, &NvLineEdit::editingFinished, this, &ExportNVHorizonDialog::newNameChanged);
	connect(buttonBox, &QDialogButtonBox::accepted, this, &ExportNVHorizonDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &ExportNVHorizonDialog::reject);
}

ExportNVHorizonDialog::~ExportNVHorizonDialog() {

}

void ExportNVHorizonDialog::accept() {
	QList<QListWidgetItem*> selectedAttributs = m_attributsListWidget->selectedItems();
	bool valid = (m_selectedItem>=0 && m_selectedItem<m_targetHorizonList->count()) ||
				(!m_newSismageHorizonName.isEmpty() && !m_newSismageHorizonName.isNull());

	QString errMsg;
	bool printErrorMsg = true;
	if (!valid) {
		errMsg = "No save name selected";
	}
	if (valid && selectedAttributs.size()<=0) {
		valid = false;
		errMsg = "No selected attributs";
	}
	if (valid && !checkNameAndAttributsCompatibility()) {
		valid = false;
		errMsg = "There is no isochron but trying to export an attribut.";
	}

	if (valid && m_selectedItem>=0 &&  m_selectedItem<m_targetHorizonList->count()) {
		QVariant var = m_targetHorizonList->item(m_selectedItem)->data(Qt::UserRole);
		valid = var.isValid() && var.toBool();

		if (!valid) {
			errMsg = "Sismage horizon not compatible. Either the format is not supported or there is a Time/Depth issue.";
		}
	}

	if (valid) {
		bool rightIssue = false;
		bool askOverwrite = false;
		bool existNotFile = false;
		bool createForbidden = false;
		QStringList badFileList;
		QStringList rightIssueList;
		QStringList overwriteList;
		QStringList existNotFileList;
		QStringList createForbiddenList;
		for (long i=0; i<selectedAttributs.size(); i++) {
			QString file = getFilePath(selectedAttributs[i]->text());
			FreeHorizon::Attribut attribut = m_freeHorizon->getLayer(selectedAttributs[i]->text());
			FixedLayerImplFreeHorizonFromDatasetAndCube* hiddenData = attribut.getFixedLayersImplFreeHorizonFromDatasetAndCube();
			valid = !file.isEmpty() && !file.isNull() && hiddenData!=nullptr;
			if (!valid) {
				badFileList << selectedAttributs[i]->text();
			} else {
				QFileInfo fileInfo(file);
				bool currentAskOverwrite = fileInfo.exists();
				if (currentAskOverwrite) {
					bool currentExistNotFile = !fileInfo.isFile();
					bool currentRightIssue = !fileInfo.isWritable();
					if (currentExistNotFile) {
						existNotFile = true;
						existNotFileList << selectedAttributs[i]->text();
					} else if (currentRightIssue) {
						rightIssue = true;
						rightIssueList << selectedAttributs[i]->text();
					}
					askOverwrite = true;
					overwriteList << selectedAttributs[i]->text();
				} else {
					QFileInfo parentDirInfo = QFileInfo(fileInfo.absolutePath());
					while (!parentDirInfo.exists() && !parentDirInfo.isRoot()) {
						parentDirInfo = QFileInfo(parentDirInfo.absolutePath());
					}
					bool currentCreateForbidden = !parentDirInfo.isWritable() || !parentDirInfo.isExecutable();
					if (currentCreateForbidden) {
						createForbidden = true;
						if (!createForbiddenList.contains(parentDirInfo.absoluteFilePath())) {
							createForbiddenList << parentDirInfo.absoluteFilePath();
						}
					}
				}
			}
		}
		if (!valid) {
			errMsg = "No save path for attributs : " + badFileList.join(", ");
		} else if (existNotFile) {
			valid = false;
			errMsg = "Expecting a file but got a folder : " + existNotFileList.join(", ");
		} else if (rightIssue) {
			valid = false;
			errMsg = "Need to overwrite a file but do not have permission : " + rightIssueList.join(", ");
		} else if (createForbidden) {
			valid = false;
			errMsg = "Folders with bad permissions : " + createForbiddenList.join(", ");
		} else if (askOverwrite) {
			QString okItem = "Overwrite";
			QString noItem = "Abort";
			QStringList options;
			options << noItem << okItem;
			QString text = QInputDialog::getItem(this, "Ask overwrite permissions", "Some files already exist : "+overwriteList.join(", ")+". Do you want to : ", options, 0, false);
			valid = text==okItem;
			if (!valid) {
				printErrorMsg = false;
				errMsg = "";
			}
		}
	}
	if (valid) {
		run();
		QDialog::accept();
	} else if (!valid && printErrorMsg) {
		QMessageBox::information(this, "Error occurred", errMsg);
	}
}

bool ExportNVHorizonDialog::checkNameAndAttributsCompatibility() {
	if (m_newSismageHorizonName.isEmpty() || m_newSismageHorizonName.isNull()) {
		return true;
	}

	QList<QListWidgetItem*> selectedAttributs = m_attributsListWidget->selectedItems();
	int i=0;
	while (i<selectedAttributs.size() && selectedAttributs[i]->text()!="isochrone") {
		i++;
	}
	if (i<selectedAttributs.size()) {
		return true;
	}
	QString file = getFilePath("isochrone");
	bool compatible = QFileInfo(file).exists();
	return compatible;
}

QListWidget* ExportNVHorizonDialog::createAttributsList() {
	QListWidget* listWidget = new QListWidget;
	listWidget->setSelectionMode(QAbstractItemView::MultiSelection);

	for (long i=0; i<m_freeHorizon->numAttributs(); i++) {
		QString attrType = QString::fromStdString(FreeHorizonManager::typeFromAttributName(m_freeHorizon->attribut(i).name().toStdString()));
		if (attrType=="isochrone" || attrType=="mean") {
			FreeHorizon::Attribut attr = m_freeHorizon->attribut(i);
			listWidget->addItem(attr.name());
		}
	}

	return listWidget;
}

QListWidget* ExportNVHorizonDialog::createTargetHorizonList() {
	QDir dir(m_sismageHorizonPath);
	QFileInfoList entries = dir.entryInfoList(QStringList() << "*.iso", QDir::Files, QDir::Name | QDir::IgnoreCase);

	QListWidget* listWidget = new QListWidget;
	listWidget->setIconSize(QSize(8, 8));
	for (long i=0; i<entries.size(); i++) {
		QListWidgetItem* item = new QListWidgetItem(entries[i].completeBaseName());
		bool itemValid = testSismageHorizon(entries[i].absoluteFilePath());
		item->setData(Qt::UserRole, itemValid);
		if (!itemValid) {
			item->setData(Qt::DecorationRole, QColor(Qt::red));
			item->setToolTip("Bad z axis type or bad format");
		}
		listWidget->addItem(item);
	}
	return listWidget;
}

QString ExportNVHorizonDialog::getFilePath(const QString& attrName) {
	QString filePath;
	QString attrType = QString::fromStdString(FreeHorizonManager::typeFromAttributName(attrName.toStdString()));
	if (attrType=="isochrone") {
		filePath = m_sismageHorizonPath + "/" + getHorizonName() + ".iso";
	} else if (attrType=="mean") {
		filePath = m_sismageHorizonPath + "/" + attrName + "." + getHorizonName() + ".amp";
	}
	return filePath;
}

QString ExportNVHorizonDialog::getHorizonName() {
	QString name;
	if (!m_newSismageHorizonName.isNull() && !m_newSismageHorizonName.isEmpty()) {
		name = m_newSismageHorizonName;
	} else if (m_selectedItem>=0 && m_selectedItem<m_targetHorizonList->count()) {
		name = m_targetHorizonList->item(m_selectedItem)->text();
	}
	return name;
}

void ExportNVHorizonDialog::horizonDestroyed() {
	deleteLater();
}

void ExportNVHorizonDialog::newNameChanged() {
	if (!m_newNameLineEdit->text().isNull() && !m_newNameLineEdit->text().isEmpty()) {
		QList<QListWidgetItem*> selectedItems = m_targetHorizonList->selectedItems();
		for (QListWidgetItem* item : selectedItems) {
			item->setSelected(false);
		}
		m_selectedItem = -1;

		m_newSismageHorizonName = m_newNameLineEdit->text();
	}
}

void ExportNVHorizonDialog::slotSelectItem(QListWidgetItem * current, QListWidgetItem * previous) {
	int count = m_targetHorizonList->count();
	for(int index = 0; index < count; index++){
		QListWidgetItem * item = m_targetHorizonList->item(index);
		QFont font = item->font();
		font.setBold(false);
		item->setFont(font);
	}

	m_selectedItem = m_targetHorizonList->currentRow();
	if (current!=nullptr) {
		QListWidgetItem* pTemp = m_targetHorizonList->currentItem();
		QFont font = pTemp->font();
		font.setBold(true);
		pTemp->setFont(font);

		m_newNameLineEdit->setText("");
		m_newSismageHorizonName = "";
	} else {
		m_selectedItem = -1;
	}
}

void ExportNVHorizonDialog::run() {
	QList<QListWidgetItem*> selectedAttributs = m_attributsListWidget->selectedItems();
	Isochron isochron(getHorizonName().toStdString(), m_surveyPath);

	for (long i=0; i<selectedAttributs.size(); i++) {
		FreeHorizon::Attribut attribut = m_freeHorizon->getLayer(selectedAttributs[i]->text());

		FixedLayerImplFreeHorizonFromDatasetAndCube* hiddenData = attribut.getFixedLayersImplFreeHorizonFromDatasetAndCube();

		if (hiddenData!=nullptr) {
			QByteArray isoBuf, attrBuf;
			hiddenData->getImageForIndex(attribut.currentImageIndex(), attrBuf, isoBuf);

			CPUImagePaletteHolder* attrHolder = new CPUImagePaletteHolder(attribut.width(), attribut.depth(), ImageFormats::QSampleType::INT16);
			attrHolder->updateTexture(attrBuf, false);

			QString attrType = QString::fromStdString(FreeHorizonManager::typeFromAttributName(selectedAttributs[i]->text().toStdString()));
			if (attrType=="isochrone") {
				isochron.saveInto(attrHolder, m_freeHorizon->cubeSeismicAddon(), true);
			} else {
				IsochronAttribut isochronAttribut(isochron, attribut.name().toStdString());
				isochronAttribut.saveInto(attrHolder, m_freeHorizon->cubeSeismicAddon(), true);
			}
		}
	}
}

bool ExportNVHorizonDialog::testSismageHorizon(const QString& sismageHorizonPath) {
	SampleUnit nvSampleUnit = m_freeHorizon->cubeSeismicAddon().getSampleUnit();
	SampleUnit sismageSampleUnit = SampleUnit::NONE;
	if (Isochron::isXtHorizon(sismageHorizonPath.toStdString())) {
		sismageSampleUnit = Isochron::getSampleUnit(sismageHorizonPath.toStdString());
	}

	return nvSampleUnit!=SampleUnit::NONE && sismageSampleUnit==nvSampleUnit;
}
