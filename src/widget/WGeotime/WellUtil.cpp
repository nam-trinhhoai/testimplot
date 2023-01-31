
#include <QString>
#include <WellUtil.h>

//void WellUtil::getIndexFromWellWellbore(const std::vector<WELLHEADDATA>& data, QString txt, int *idx_well, int *idx_bore)
//{
//	*idx_well = -1;
//	*idx_bore = -1;
//
//	for (int iw=0; iw<data.size(); iw++)
//	{
//		for (int ib=0; ib<data[iw].bore.size(); ib++)
//		{
//			QString str0 = data[iw].bore[ib].tinyName;
//			if ( str0.compare(txt) == 0 )
//			{
//				*idx_well = iw;
//				*idx_bore = ib;
//				return;
//			}
//		}
//	}
//}



void WellUtil::getIndexFromWellBoreFullname(const std::vector<WELLHEADDATA>& data, QString fullname, int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;

	for (int n=0; n<data.size(); n++)
	{
		for (int m=0; m<data[n].bore.size(); m++)
		{

				if ( data[n].bore[m].fullName.compare(fullname) == 0 )
				{
					*idx_well = n;
					*idx_bore = m;
					return;
				}
		}
	}
}

bool WellUtil::getIndexFromWellLists(const std::vector<WELLLIST>& data, QString fullname, int *idx_well, int *idx_bore) {
	*idx_well = -1;
	*idx_bore = -1;

	for (int n=0; n<data.size(); n++)
	{
		for (int m=0; m<data[n].wellborelist.size(); m++)
		{
			if ( data[n].wellborelist[m].bore_fullname.compare(fullname) == 0 )
			{
				*idx_well = n;
				*idx_bore = m;
				return true;
			}
		}
	}
	return false;
}


void WellUtil::wellListCreateGetIndex(std::vector<WELLLIST> *well_list, QString head_tinyname, QString head_fullname,
		QString bore_tinyname, QString bore_fullname,
		QString deviation_fullname,
		int *idx_well, int *idx_bore)
{
	*idx_well = -1;
	*idx_bore = -1;
	int idx0 = -1, n = 0;
	while ( n < well_list->size() && idx0 < 0 )
	{
		// qDebug() << well_list[n].head_fullname << head_fullname;

		if ( (*well_list)[n].head_fullname.compare(head_fullname) == 0 )
		{
			idx0 = n;
		}
		n++;
	}

	if ( idx0 < 0 )
	{
		WELLLIST welllist;
		well_list->push_back(welllist);
		int n0 = well_list->size()-1;
		(*well_list)[n0].head_tinyname = head_tinyname;
		(*well_list)[n0].head_fullname = head_fullname;
		*idx_well = n0;
	}
	else
	{
		*idx_well = idx0;
	}

	idx0 = -1, n = 0;
	while ( n < (*well_list)[*idx_well].wellborelist.size() && idx0 < 0 )
	{
		if ( (*well_list)[*idx_well].wellborelist[n].bore_fullname.compare(bore_fullname) == 0 )
		{
			idx0 = n;
		}
		n++;
	}
	if ( idx0 < 0 )
	{
		WELLBORELIST wellborelist;
		(*well_list)[*idx_well].wellborelist.push_back(wellborelist);
		int n0 = (*well_list)[*idx_well].wellborelist.size()-1;
		(*well_list)[*idx_well].wellborelist[n0].bore_tinyname = bore_tinyname;
		(*well_list)[*idx_well].wellborelist[n0].bore_fullname = bore_fullname;
		(*well_list)[*idx_well].wellborelist[n0].deviation_fullname = deviation_fullname;
		*idx_bore = n0;
	}
	else
	{
		*idx_bore = idx0;
	}
}




std::vector<std::vector<QString>> WellUtil::multiCriterionSearchFormat(const QString& str)
{
	QStringList list0 = str.split(";",Qt::SkipEmptyParts);

	std::vector<std::vector<QString>> listf;
	listf.resize(4);

	for (int n=0; n<list0.size(); n++)
	{
		QStringList list = list0[n].split("=", Qt::SkipEmptyParts);
		if ( list.size() == 1 )
		{
			listf[0].push_back(list[0]);
		}
		else
		{
			QStringList listx = list[1].split(" ", Qt::SkipEmptyParts);
			if ( list[0].compare("log") == 0  )
			{
				for (int p=0; p<listx.size(); p++)
				{
					listf[1].push_back(listx[p]);
				}
			}
			else if ( list[0].compare("tf2p") == 0 )
			{
				for (int p=0; p<listx.size(); p++)
				{
					listf[2].push_back(listx[p]);
				}
			}
			else if ( list[0].compare("picks") == 0 )
			{
				for (int p=0; p<listx.size(); p++)
				{
					listf[3].push_back(listx[p]);
				}
			}
		}
	}
	return listf;
}


int WellUtil::qstring_display_listPrefix_valid(QString str, std::vector<QString> prefix)
{
	int nbsearch = prefix.size();
	if ( nbsearch == 0 ) return 1;
	int val = 0;
	for ( int s=0; s<nbsearch; s++)
	{
		int idx = str.indexOf(prefix[s], 0, Qt::CaseInsensitive);
	    if ( idx >= 0 ) val ++;
	}
	// if ( val == nbsearch || nbsearch == 0 ) return 1;
	if ( val > 0 ) return 1;
	return 0;
}


int WellUtil::well_wellbore_display_valid(const std::vector<WELLHEADDATA>& data0, const QString& prefix, int i_well, int i_bore)
{
	if ( prefix.compare("") == 0 ) return 1;

	std::vector<std::vector<QString>> list_prefix = multiCriterionSearchFormat(prefix);
	std::vector<QString> t_tiny;

	int ret0 = 0, ret1 = 0, ret2 = 0, ret3 = 0;

	int ret = qstring_display_listPrefix_valid(data0[i_well].bore[i_bore].tinyName, list_prefix[0]);
	if ( ret == 1 ) ret0 = 1;

	t_tiny = data0[i_well].bore[i_bore].logs.getTiny();
	if ( t_tiny.size() == 0 && list_prefix[1].size() == 0 )
		ret1 = 1;
	else
		for (int i=0; i<t_tiny.size(); i++ )
		{
			int ret = qstring_display_listPrefix_valid(t_tiny[i], list_prefix[1]);
			if ( ret == 1 ) ret1 = 1;
		}

	t_tiny = data0[i_well].bore[i_bore].tf2p.getTiny();
	if ( t_tiny.size() == 0 && list_prefix[2].size() == 0 )
		ret2 = 1;
	else
		for (int i=0; i<t_tiny.size(); i++ )
		{
			int ret = qstring_display_listPrefix_valid(t_tiny[i], list_prefix[2]);
			if ( ret == 1 ) ret2 = 1;
		}

	t_tiny = data0[i_well].bore[i_bore].picks.getTiny();
	if ( t_tiny.size() == 0 && list_prefix[3].size() == 0 )
		ret3 = 1;
	else
		for (int i=0; i<t_tiny.size(); i++ )
		{
			int ret = qstring_display_listPrefix_valid(t_tiny[i], list_prefix[3]);
			if ( ret == 1 ) ret3 = 1;
		}
	int res = ret0*ret1*ret2*ret3;
	if ( res == 0 ) return 0;
	return 1;
	// return ret0*ret1*ret2*ret3;
}

