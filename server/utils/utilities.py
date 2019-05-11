from itertools import combinations

from server.models.writer import *
from server.views.writersvo import *
from server.views.profilevo import *
from server.models.features import *
from server.httpresponses.errors import *
from server.httpresponses.messages import *
from faker import Faker
import re
import datetime


def writer_to_dict(writer):
    """
    Convert writer object into dictionary
    :param writer: writer model
    :return: writer_dict: dictionary containing writer's info
    """
    writer_features_dict = writer.features.__dict__
    features = {'_features': writer_features_dict}
    writer_dict = writer.__dict__
    writer_dict.update(features)

    return writer_dict


def dict_to_writer(writer_dict):
    """
    Convert dictionary to writer object
    :param writer_dict:
    :return: writer model (object)
    """
    writer = Writer()
    writer.id = writer_dict["_id"]
    writer.name = writer_dict["_name"]
    writer.username = writer_dict["_username"]

    features_dict = writer_dict["_features"]
    features = Features()
    features.horest_features = features_dict["_horest_features"]
    features.texture_feature = features_dict["_texture_feature"]
    features.sift_SDS = features_dict["_sift_SDS"]
    features.sift_SOH = features_dict["_sift_SOH"]

    writer.features = features

    return writer


def dict_to_profile(writer_dict):
    """
    Convert writer dictionary into ProfileVo object
    :param writer_dict: dictionary contains writer's attributes
    :return: ProfileVo object
    """
    profile = ProfileVo()
    profile.id = writer_dict["_id"]
    profile.name = writer_dict["_name"]
    profile.username = writer_dict["_username"]
    profile.address = writer_dict["_address"]
    profile.phone = writer_dict["_phone"]
    profile.nid = writer_dict["_nid"]
    profile.image = writer_dict["_image"]
    profile.birthday = writer_dict["_birthday"]

    return profile


def dict_to_writers(writer_dict):
    """
    Convert writer dictionary into writers model
    :param writer_dict: dictionary returned from database
    :return: writer object of writers model
    """
    writer = WritersVo()
    writer.id = writer_dict["_id"]
    writer.name = writer_dict["_name"]
    writer.username = writer_dict["_username"]

    return writer


def validate_writer_request(request):
    """
    Validate the create writer request regarding phone format, and national ID format
    :param request:
    :return: HTTP Error code:
                - 200 for success
                - 400 if validation failed
             HTTP Message:
                - "OK" for success
                - "Phone is invalid" for invalid phone format
                - "National ID is invalid" for invalid National ID format
    """
    phone_pattern = re.compile("(01)[0 1 2 5][0-9]{8}")
    match_phone = phone_pattern.match(request["_phone"])

    nid_pattern = re.compile("(2|3)[0-9][1-9][0-1][1-9][0-3][1-9](01|02|03|04|11|12|13|14|15|16|17|18|19|21|22|23|24|25|26|27|28|29|31|32|33|34|35|88)\d\d\d\d\d")
    match_nid = nid_pattern.match(request["_nid"])

    if match_phone is None:
        return HttpErrors.BADREQUEST, HttpMessages.INVALIDPHONE
    elif match_nid is None:
        return HttpErrors.BADREQUEST, HttpMessages.INVALIDNID
    else:
        return HttpErrors.SUCCESS, HttpMessages.SUCCESS


def func(writer):
    """
    Function to return attribute writer id
    :param writer: writer model object
    :return: attribute id
    """
    return writer.id


def fake_data():
    """
    Function to fake writer's data
    :return: names, birthdays, phones, addresses, nid
    """
    fake = Faker()

    names = ["Abdul Ahad", "Abdul Ali", "Abdul Alim", "Abdul Azim", "Abu Abdullah", "Abu Hamza", "Ahmed Tijani", "Ali Reza",
             "Aman Ali", "Anisur Rahman", "Azizur Rahman", "Badr al-Din", "Baha' al-Din", "Barkat Ali", "Burhan al-Din", "Fakhr al-Din",
             "Fazl UrRahman", "Fazlul Karim", "Fazlul Haq", "Ghulam Faruq", "Ghiyath al-Din", "Ghulam Mohiuddin", "Habib ElRahman", "Hamid al-Din",
             "Hibat Allah", "Husam ad-Din", "Ikhtiyar al-Din", "Imad al-Din", "Izz al-Din", "Jalal ad-Din", "Jamal ad-Din", "Kamal ad-Din",
             "Lutfur Rahman", "Mizanur Rahman", "Mohammad Taqi", "Nasir al-Din", "Seif ilislam", "Sadr al-Din", "Sddam Hussein", "Samar Gamal",
             "May Ahmed", "Ahmed Khairy", "Omar Ali", "Salma Ibrahim", "Ahmed Gamal", "Hadeer Hossam", "Hanaa Ahmed", "Gamal Saad",
             "Bisa Dewidar", "Ahmed Said", "Nachwa Ahmed", "Ezz Farhan", "Nourhan Farhan", "Mariam Farhan", "Mouhab Farhan", "Sherif Ahmed",
             "Noha Ahmed", "Yasmine Sherif", "Eslam Sherif", "Ahmed Sherif", "Mohamed Ahmed", "Zeinab Khairy", "Khaled Ali", "Rana Ali", "Ali Shaalan", "Ahmed Youssry",
             "AbdelRahman Nasser", "Youssra Hussein", "Ingy Alaa", "Rana Afifi", "Nour Attya", "Amani Tarek", "Salma Ahmed", "Iman Fouad", "Karim ElRashidy", "Ziad Mansour",
             "Mohamed Salah", "Anas ElShazly", "Hazem Aly", "Youssef Maraghy", "Ebram Hossam", "Mohamed Nour", "Mohamed Ossama", "Hussein Hosny",
             "Ahmed Samy", "Youmna Helmy", "Kareem Haggag", "Nour Yasser", "Farah Mohamed", "Ahmed Hisham", "Omar Nashaat", "Mohamed Yasser",
             "Sara Hassan", "Ahmed keraidy", "Magdy Hafez", "Waleed Mostafa", "Khaled Hesham", "Karim Hossam", "Omar Nasharty", "Rayhana Ayman"]

    addresses = ["36 El Salam St. - El Saada City - Shoubra El Khima - Cairo - Egypt",
                 "Mahdi St., Azbakiya, Cairo",
                 "125 Shoubra St. -Cairo - Egypt",
                 "67 Mohamad El Nadi Street, Cairo Egypt",
                 "12 Sherif Basha El Kbeer ST, Abdeen",
                 "7 Zohyt Esamt St, From Ein Shams St 11131, Cairo",
                 "100-El Moaz Ledin Elah EL Famtamy, EL Gamlia St., Cairo",
                 "16 Iran St, Dokki, Cairo",
                 "9 Merrit St., in front of Egyptian Museum; Tahrir",
                 "17 Beirut, Heliopolis, Cairo",
                 "Back Of Cairo Airport, El-Salam City. Cairo., Cairo",
                 "18 Egyptian Saudi Finance Bank, El-Nouzha St.",
                 "Back Of Cairo Airport, El-Salam City. Cairo., Cairo",
                 "Heliopolis, Cairo, Cairo",
                 "34 Oraby St. - El Nahada Square Saraiat El Maadi -  Cairo - Egypt",
                 "Shbramant - Giza - Egypt",
                 "El-Gezeira El-Wasta, Egypt Tower",
                 "Zalika  Bata",
                 "Elqanto Sq. Elmoski Cairo - 56, Saad Zaghlol St. Portsaid., cairo",
                 "13 Howd El Gabal Industrial Zone - Cairo - Egypt",
                 "7 El Gezawi St. - Industrial Zone - Shoubra El Khima - El Kaliubia - Egypt",
                 "100-El Moaz Ledin Elah EL Famtamy, EL Gamlia St., Cairo",
                 "26 adly st, ciro egypt",
                 "56 Abdel El Khalek Tharwat, Downtown, cairo",
                 "10 Mohamed Wahid St.- From Bigam Road -Shoubra El Khima - El Kaliubia  - Egypt",
                 "15 Ahmed Mohamed Kamal St. Behindegypt Air In Abas EL Akad/Nasr City/Cairo",
                 "115 Manial Street, Cairo, Cairo",
                 "Al Emdad Wa Altamween Bldgs., Al Nasr Road, Cairo",
                 "sakr korish buildings, New Maadi, cairo",
                 "26 adly st, ciro egypt",
                 "100-El Moaz Ledin Elah EL Famtamy, EL Gamlia St., Cairo",
                 "10 Mohamed Wahid St.- From Bigam Road -Shoubra El Khima - El Kaliubia  - Egypt",
                 "24 Menouf Street, Heliopolis",
                 "6 Octobr, Cairo, Cairo",
                 "Mahdi St., Azbakiya, Cairo",
                 "Mohammed Hafiz St., Mohande Cairo, Cairo",
                 "7 El Gezawi St. - Industrial Zone - Shoubra El Khima - El Kaliubia - Egypt",
                 "58 Ahmed Orabi St., Mohandseen, Cairo",
                 "80 Mosadek, St, Cairo",
                 "8 El-Nozha St.; Egyptian Saudi Construction Co. Bldg.",
                 "85 Saudi Egyptian Construction Co. Bldgs. - Behind Qubba Pal",
                 "3 Shaaker El Gendi St.- El Sharabia - Cairo - Egypt",
                 "Saint Fatima, Heliopolis, Cairo",
                 "National Bank of Egypt Bldg 23 ElRoda St, Mamalik Sq.",
                 "10 Mohamed Wahid St.- From Bigam Road -Shoubra El Khima - El Kaliubia  - Egypt",
                 "9 Merrit St., in front of Egyptian Museum; Tahrir",
                 "Mit Halfa - Behind Seliman Baktar -El Kaliuobia - Egypt",
                 "El Nozha Str., Nasr City, Cairo",
                 "36 Ard El Awkaf Hadaek El Kobba, cairo",
                 "Klm. 25 Misr Esmalia Desert Road - El Abbour City - Cairo - Egypt",
                 "11 Ahmed Mokhtar Hegazy St. Manial",
                 "14 Zaki Elmohandess Elnozha Elgedda",
                 "Saint Fatima, Heliopolis, Cairo",
                 "228 Elhorria Street, Alex",
                 "International Road, Abu Yossief, Alex",
                 "56 ismail mehnna",
                 "Eltabia Alex Egypt",
                 "58 Ahmed Hafez St., Sidibesher, alex",
                 "60 Elgaish Road, Elebrahimia",
                 "Alex Agriculture Rd.",
                 "8a EL Mamar EL Gharby Street. Tanta Egypt",
                 "Extension Of el bahr st.",
                 "15 Ahmed Mohamed Kamal St. Behindegypt Air In Abas EL Akad/Nasr City/Cairo",
                 "St. No. 47 Industrial Zone - El AbbasiaCairo - Egypt",
                 "Beside Military Hotel, Cairo, Cairo",
                 "Beer Homos, Elmosky, Cairo",
                 "First of Mit Ghamr Road - El Sanblawin -  Egypt",
                 "Reija Street, Cairo, Cairo",
                 "Nasir City Cairo Egypt",
                 "Maadi, Cairo, Cairo",
                 "79 H Al Nasr Road, Nasr City, Cairo",
                 "National Bank of Egypt Bldg 23 ElRoda St, Mamalik Sq.",
                 "Behind Nile T. V, Mokattam, Cairo",
                 "6 makhlouf St, Dokky Guiza. Apt. No7, Cairo",
                 "110 Kasarat  ElBaladia St. - El Kosarin -El Zawia El Hamra Cairo - Egypt",
                 "115 Manial Street, Cairo, Cairo",
                 "13A Awlad Emara St. -Bigam Road - El Kanater - El Kliuobia - Egypt",
                 "Industrial Area 1, Plot No 8/9, Cairo"]

    for i in range(78, 100):
        addresses.append(fake.address())

    birthdays = []
    for i in range(0, 100):
        birthdays.append(fake.date(pattern="%Y-%m-%d", end_datetime=datetime.date(2000, 12, 12)))

    phones = ["01116055544", "01018028105", "01270625965", "01277217139", "01253194815",
              "01231867718", "01015120204", "01025120204", "01002191561", "01012226165",
              "01014890440", "01055351750", "01050723791", "01002176931", "01219310301",
              "01236664582", "01052918385", "01293917862", "01111661540", "01111661541",
              "01111661542", "01111661543", "01111661544", "01111661545", "01111661546"]

    for i in range(0, 25):
        phones.append(phones[i][0:5] + '1'+ phones[i][6:])
        phones.append(phones[i][0:5] + '2'+ phones[i][6:])
        phones.append(phones[i][0:5] + '3'+ phones[i][6:])


    nid = ["29612342381123"]
    combination = list(combinations(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 5))
    for i in range(0, 99):
        nid.append(nid[0][0:9] + ''.join(combination[i]))

    return names, birthdays, phones, addresses, nid
