#ifndef SURVEYSELECTIONDIALOG_H
#define SURVEYSELECTIONDIALOG_H

#include <QDialog>
#include <QString>
#include <QVector>
#include <QTabWidget>
#include <QListWidget>

namespace Ui {
class SurveySelectionDialog;
}

class SurveySelectionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SurveySelectionDialog(QWidget *parent = 0);
    ~SurveySelectionDialog();

    QString getDirProject();
    QString getProject();

private slots:
    void setDirProject(int index);
    void setProject(const QString& project);
    void setProjectSlot(QListWidget* listWidget);

private:
    void clearProjectsList();
    void loadDirProjects();


    Ui::SurveySelectionDialog *m_ui;
    QTabWidget* m_projectsTabWidget = nullptr;

    QVector<std::pair<QString, QString>> m_dirProjects;

    QMap<QChar, QStringList> m_mapCharToProjects;
    QMap<QChar, QListWidget*> m_projectsLists;
    QString m_currentProjectDir;
    QString m_currentProject;

};

#endif // SURVEYSELECTIONDIALOG_H
