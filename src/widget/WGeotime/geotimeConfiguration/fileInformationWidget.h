
#ifndef __FILEINFORMATIONWIDGET__
#define __FILEINFORMATIONWIDGET__


#include <QDialog>
#include <QtGui>
#include <QPlainTextEdit>

#include <vector>
#include <QStringList>

class QListWidget;
//class QStringList;
class QString;
class QListWidgetItem;
class QLineEdit;
class QComboBox;


class FileInformationWidget :public QDialog{
    Q_OBJECT
  public:
	FileInformationWidget(QString filename);
    virtual ~FileInformationWidget();
    // // static void infoFromFilename(Q_OBJECT *parent, QString filename);
    static QString infoFromFilename(QString filename);
    static QString getFormatedFileSize(QString filename);



private:
    QPlainTextEdit *textInfo = nullptr;
    void infoDisplay(QString filename);




};



#endif
