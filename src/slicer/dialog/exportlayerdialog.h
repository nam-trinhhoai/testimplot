/*
 * exportlayerdialog.h
 *
 *  Created on: Jan 14, 2021
 *      Author: l0222891
 */

#include <QDialog>
#include <QtGui>
#include <QStringList>

class QListWidget;
//class QStringList;
class QString;
class QListWidgetItem;
class QLineEdit;
class QRegularExpressionValidator;

#ifndef TARUMAPP_SRC_DIALOG_EXPORTLAYERDIALOG_H_
#define TARUMAPP_SRC_DIALOG_EXPORTLAYERDIALOG_H_

class ExportLayerDialog :public QDialog{
    Q_OBJECT
  public:
	ExportLayerDialog(const QStringList& pList, QString const& title);
    virtual ~ExportLayerDialog();
    bool isNewName() const;
    QString newName() const;
    int getSelectedIndex() const;
    QString getSelectedString() const;

  private slots:
  	  void slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous);
  	  void newNameChanged();

  private:
    const QStringList& m_stringList;
    QListWidget* m_listWidget;
    QLineEdit* m_lineEdit;
    QRegularExpressionValidator* m_validator;

    int m_selectedItem = -1;
    QString m_newName;
};

#endif /* TARUMAPP_SRC_DIALOG_EXPORTLAYERDIALOG_H_ */
