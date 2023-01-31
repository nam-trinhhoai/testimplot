
#ifndef __FILESELECTORDIALOG__
#define __FILESELECTORDIALOG__

#include <QDialog>
#include <QtGui>

#include <vector>
#include <QStringList>

class QListWidget;
//class QStringList;
class QString;
class QListWidgetItem;
class QLineEdit;
class QComboBox;


class FileSelectorDialog :public QDialog{
    Q_OBJECT
  public:
	enum MAIN_SEARCH_LABEL { all=0, seismic, dip, dipxy, dipxz, rgt, patch, rgb2, Avi, horizon };
	FileSelectorDialog(const std::vector<QString>* pList, QString const& title);
    virtual ~FileSelectorDialog();
    int getSelectedIndex() const;
    std::vector<int> getMultipleSelectedIndex() const;
    QString getSelectedString() const;
    void setMainSearchType(int val);
    void setMultipleSelection(bool val);
    void setDataPath(std::vector<QString>* pData);


  private slots:
  	  void slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous);
  	  void listFileDoubleClick(QListWidgetItem* item);
  	  void trt_SearchChange(QString txt);
  	  void trt_mainChangeDisplay(int idx);

  private:
  	std::vector<QString> MAIN_SEARCH_PREFIX = { "all", "seismic", "dip", "dipxy", "dipxz", "rgt", "patch", "rgb2", "avi", "horizon"};
  	std::vector<QString> FILE_SORT_PREFIX = { "rgt", "dipxy", "dipxz", "__nextvisionpatch"};
  	std::vector<QString> RGT_SORT = {"rgt"};
  	std::vector<QString> DIP_SORT = {"dipxy", "dipxz"};
  	std::vector<QString> DIPXY_SORT = {"dipxy"};
  	std::vector<QString> DIPXZ_SORT = {"dipxz"};
  	std::vector<QString> PATCH_SORT = {"__nextvision"};
  	std::vector<QString> AVI_SORT = {"video"};
    const std::vector<QString> *m_list0;
    const std::vector<QString> *m_path = nullptr;
    QListWidget* m_listWidget;
    QLineEdit *m_searchString;
    QComboBox *m_mainSearch;
    int m_mainSearchType = MAIN_SEARCH_LABEL::all;

    int m_selectedItem = -1;
    bool isMultiKeyInside(QString str, QString key);
    std::vector<QString> getMainSearchAllFiles();
    std::vector<QString> getMainSearchSeismicFiles();
    std::vector<QString> getMainSearchSpecificFiles(std::vector<QString> key);
    std::vector<QString> getMainSearchFiles();
    void displayNames();
    QString getPathFromName(QString name);
};


#endif
