/*
 * StringSelectorDialog.h
 *
 *  Created on: Apr 6, 2020
 *      Author: l0222891
 */

#include <QDialog>
#include <QtGui>
#include <QStringList>

class QListWidget;
//class QStringList;
class QString;
class QListWidgetItem;

#ifndef TARUMAPP_SRC_DIALOG_STRINGSELECTORDIALOG_H_
#define TARUMAPP_SRC_DIALOG_STRINGSELECTORDIALOG_H_

class StringSelectorDialog :public QDialog{
    Q_OBJECT
  public:
    StringSelectorDialog(QStringList* pList, QString const& title);
    virtual ~StringSelectorDialog();
    int getSelectedIndex() const;
    QString getSelectedString() const;

  private slots:
  	  void slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous);

  private:
    QStringList* m_stringList;
    QListWidget* m_listWidget;

    int m_selectedItem = -1;
};

#endif /* TARUMAPP_SRC_DIALOG_STRINGSELECTORDIALOG_H_ */
