#ifndef MANAGERUPDATEWIDGET_H
#define MANAGERUPDATEWIDGET_H

#include <QWidget>
#include <QString>
#include <QListWidget>
#include <QTreeWidget>
#include <QTabWidget>
#include <QGroupBox>
#include <QPushButton>
#include <QLineEdit>
#include <QStringList>
#include "workingsetmanager.h"
#include <vector>
#include "WellUtil.h"
#include "GeotimeProjectManagerWidget.h"

class ManagerUpdateWidget : public QWidget{
    Q_OBJECT
public:
    ManagerUpdateWidget(QString dataName,WorkingSetManager *manager,QWidget* parent = 0);
    virtual ~ManagerUpdateWidget();
    const std::vector<QString>& getDataTinyName(){return m_SelectedDataTinyname;}
    const std::vector<QString>& getDataFullName(){return m_SelectedDataFullname;}
    const std::vector<WELLLIST>&  getWellList() {return m_wellList;}
    std::vector<MARKER> getPicksList();
    const std::vector<QString>&  getSelectedNurbsName() {return m_selectedNurbsName;}
    const std::vector<QString>&  getSelectedNurbsFullname() {return m_selectedNurbsFullname;}

    bool forceAllItems() const;
    void setForceAllItems(bool val);

public slots:
//   void updateData(QString text);
   void trt_SearchChange(QString text);
   void dataChanged(const QModelIndex &topLeft,const QModelIndex &bottomRight, const QVector<int> &roles);

private:
   void dataSeismicGui(WorkingSetManager *manager);
   void dataNextVisionHorizonGui(WorkingSetManager *manager);
   void dataWellsGui(WorkingSetManager *manager);
   void dataIsoHorizonGui(WorkingSetManager *manager);
   void dataNurbsGui(WorkingSetManager *manager);
   void dataPicksGui(WorkingSetManager *manager);

    void select_data(QString &rstrItem);
    void unselect_data(QString &rstrItem);
    void displayWellsDataTree(QString prefix = "");
    void display_data_tree(QString prefix="");
    void trace();


    void addPick(QString &rstrItem);
    void deletePick(QString &rstrItem);
    void addWell(int wellIdx,int indexBore);
    void deleteWell(QString &rstrItem);
    void updateWells(QString &rstrItem);
    bool checkWellManager(QString &rstrItem);
    QStringList spitSearchItem(QString line);
    bool isDiplayName(QString& name, QStringList& list);
    bool isDiplayName(std::vector<QString>& names, QStringList& list);
    std::vector<QString> getHorizonInTree();
    bool isNameExist(QString name, std::vector<QString> list);

  //  void addNurbsName(QString name);

    QLineEdit *pLineEditSearch = nullptr;
    QLineEdit *pLineEditSearchWellLog = nullptr;
    QLineEdit *pLineEditSearchWellTf2p = nullptr;
    QLineEdit *pLineEditSearchWellPicks = nullptr;
    QString n_name;
    QTreeWidget *m_Data_SelectionTree;
    QTabWidget *m_tab_widget;
    WorkingSetManager *m_manager;
    std::vector<QString> m_listData;
    std::vector<QString> m_DataTinyname;
    std::vector<QString> m_DataFullname;
    std::vector<QString> m_SelectedDataTinyname;
    std::vector<QString> m_SelectedDataFullname;
    std::vector<PMANAGER_WELL_DISPLAY>  m_wellDisplayList;
    std::vector<WELLLIST>  m_wellList;
    std::vector<QString> m_allPicksNames;
    std::vector<QString> m_allPicksPaths;
    std::vector<QBrush> m_allPicksColors;
    std::vector<QString> m_picksNames;
    std::vector<QString> m_picksPaths;
    std::vector<QBrush> m_picksColors;

    std::vector<QString> m_horizonInTree;

    std::vector<QString> m_selectedNurbsName;
    std::vector<QString> m_selectedNurbsFullname;

    bool m_forceAllItems = false;
//    QString m_SelectedBore;
};


#endif // MANAGERUPDATEWIDGET_H
