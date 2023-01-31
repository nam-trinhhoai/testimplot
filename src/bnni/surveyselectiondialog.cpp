#include "surveyselectiondialog.h"
#include "ui_surveyselectiondialog.h"
#include "functionselector.h"

#include "ProjectManager.h"
#include "globalconfig.h"

#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QDebug>

SurveySelectionDialog::SurveySelectionDialog(QWidget *parent) :
    QDialog(parent),
    m_ui(new Ui::SurveySelectionDialog)
{
    m_ui->setupUi(this);

    // setup combobox
    loadDirProjects();
    connect(m_ui->projectDirComboBox, SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &SurveySelectionDialog::setDirProject);

    // setup tab widget
    m_projectsTabWidget = new QTabWidget;
    m_projectsTabWidget->setTabPosition(QTabWidget::West);
    m_ui->horizontalLayout->addWidget(m_projectsTabWidget);

    char it='0';
    QListWidget* projectsList = new QListWidget;
    projectsList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_projectsTabWidget->addTab(projectsList, QString("0-9"));
    connect(projectsList, &QListWidget::itemSelectionChanged, [this, projectsList]() {
        this->setProjectSlot(projectsList);
    });

    while (it<='9') {
        m_mapCharToProjects.insert(QChar(it), QStringList());
        m_projectsLists.insert(QChar(it), projectsList);
        it++;
    }

    it='a';
    while (it<='z') {
        m_mapCharToProjects.insert(QChar(it), QStringList());

        projectsList = new QListWidget;
        projectsList->setSelectionMode(QAbstractItemView::SingleSelection);
        m_projectsLists.insert(QChar(it), projectsList);
        m_projectsTabWidget->addTab(projectsList, QString(QChar(it)));

        connect(projectsList, &QListWidget::itemSelectionChanged,  [this, projectsList]() {
            this->setProjectSlot(projectsList);
        });
        it++;
    }

    connect(m_ui->buttonBox, &QDialogButtonBox::rejected, this, [this]() {
        reject();
    });
}

void SurveySelectionDialog::loadDirProjects() {
    m_ui->projectDirComboBox->clear();
    m_ui->projectDirComboBox->addItem(tr(""),"");
    m_dirProjects.clear();

    GlobalConfig& config = GlobalConfig::getConfig();
    const std::vector<std::pair<QString, QString>>& dirProjects = config.dirProjects();

    for (const std::pair<QString, QString>& pair : dirProjects) {
    	m_dirProjects.append(pair);
        QString projectName = pair.first;
        QString projectPath = pair.second;

        m_ui->projectDirComboBox->addItem(projectName,projectPath);
    }
}

SurveySelectionDialog::~SurveySelectionDialog()
{
    delete m_ui;
}

void SurveySelectionDialog::setDirProject(int index) {
    clearProjectsList();

    m_currentProjectDir = m_ui->projectDirComboBox->itemData(index, Qt::UserRole).toString();
    if (m_currentProjectDir.isEmpty() || m_currentProjectDir.isNull()) {
        return;
    }


    QDir projectsDir(m_currentProjectDir);
    std::vector<QString> names = ProjectManager::getListDir(projectsDir.absolutePath());

    for (int i=0; i<names.size(); i++) {
        QString baseName = names[i];
        if (baseName.isNull() || baseName.isEmpty()) {
            continue;
        }
        QChar _char = baseName.at(0).toLower();
        if (m_mapCharToProjects.contains(_char)) {
            m_mapCharToProjects[_char] << baseName;
            m_projectsLists[_char]->addItem(baseName);
        } else {
            qDebug() << "SurveySelectionDialog::setDirProject : Unexpected QChar as first string character";
        }
    }
}

void SurveySelectionDialog::clearProjectsList() {
    m_currentProject = "";

    for (QChar _char : m_mapCharToProjects.keys()) {
        m_mapCharToProjects[_char].clear();
        m_projectsLists[_char]->clear();
    }
}

void SurveySelectionDialog::setProjectSlot(QListWidget* listWidget) {
	QList<QListWidgetItem*> selection = listWidget->selectedItems();
	if (selection.count()>0) {
		setProject(selection[0]->text());
	}
}

void SurveySelectionDialog::setProject(const QString& project) {
	QDir dir(m_currentProjectDir);
	QFileInfo fileInfo(dir.absoluteFilePath(project));
    if (fileInfo.exists() && fileInfo.isDir()) {
        m_currentProject = project;
        accept();
    } else {
    	QString errorMessage;
    	if (!fileInfo.exists() && fileInfo.isSymbolicLink()) {
    		errorMessage = "You do not have access to this project. You need to contact project administrator.";
    	} else if (!fileInfo.exists()) {
    		errorMessage = "This project has been deleted during dialog execution.";
    	} else {
    		errorMessage = "This item is not a valid directory. It is a fast display leftover.";
    	}
    	QMessageBox::information(this, "Bad Project", errorMessage);

    }
}

QString SurveySelectionDialog::getDirProject() {
    return m_currentProjectDir;
}

QString SurveySelectionDialog::getProject() {
    return m_currentProject;
}
